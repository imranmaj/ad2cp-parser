# https://support.nortekgroup.com/hc/en-us/articles/360029513952-Integrators-Guide-Signature

from typing import IO, Tuple, Union, Callable, List, Optional
from enum import Enum, unique


@unique
class BurstAverageDataRecordVersion(Enum):
    VERSION2 = 2  # Burst/Average Data Record Definition (DF2)
    VERSION3 = 3  # Burst/Average Data Record Definition (DF3)


# name of the field
type_field_name = Optional[str]
# size of the field in bytes. Can be a predicate
type_field_size_bytes = Union[int, Callable[["Ad2cpDataPacket"], int]]
# whether the field is signed. If None, it is kept as bytes instead of int
type_field_signed = Optional[bool]
# predicate to determine whether the field should exist at all
type_predicate = Callable[["Ad2cpDataPacket"], bool]
type_without_predicate_field = Tuple[type_field_name,
                                     type_field_size_bytes, type_field_signed]
type_with_predicate_field = Tuple[type_field_name,
                                  type_field_size_bytes, type_field_signed, type_predicate]
type_field = Union[type_without_predicate_field, type_with_predicate_field]


class NoMorePacketsError(Exception):
    pass


class Ad2cpReader:
    def __init__(self, f: IO, data_record_format_type: BurstAverageDataRecordVersion):
        self.packets = []
        while True:
            try:
                self.packets.append(Ad2cpDataPacket(
                    f, data_record_format_type))
            except NoMorePacketsError:
                pass


class Ad2cpDataPacket:
    def __init__(self, f: IO, burst_average_data_record_version: BurstAverageDataRecordVersion):
        self.burst_average_data_record_version = burst_average_data_record_version
        self._read_header(f)
        self._read_data_record(f)

    @staticmethod
    def _read_exact(f: IO, total_num_bytes_to_read: int) -> bytes:
        """
        Repeatedly reads from a stream until EOF or the correct number of bytes is reached
        """

        all_bytes_read = bytes()
        if total_num_bytes_to_read <= 0:
            return all_bytes_read
        num_bytes_to_read_remaining = total_num_bytes_to_read
        last_bytes_read = None
        while last_bytes_read is None or (len(last_bytes_read) > 0 and len(all_bytes_read) < total_num_bytes_to_read):
            last_bytes_read = f.read(num_bytes_to_read_remaining)
            num_bytes_to_read_remaining -= len(last_bytes_read)
            if len(last_bytes_read) > 0:
                all_bytes_read += last_bytes_read
        return all_bytes_read

    def _read_header(self, f: IO):
        raw_header = self._read_data(f, self.HEADER_FORMAT)
        print("header checksum     ", self.checksum(raw_header[: -2]), self.header_checksum)
        # don't include the last 2 bytes, which is the header checksum itself
        assert self.checksum(
            raw_header[: -2]) == self.header_checksum, "invalid header checksum"

    def _read_data_record(self, f: IO):
        # TODO: figure out where to send the other ids
        if self.id in (0x15, 0x16, 0x18, ):  # burst/average
            if self.burst_average_data_record_version == BurstAverageDataRecordVersion.VERSION2:
                raw_data_record = self._read_data(f, self.BURST_AVERAGE_VERSION2_DATA_RECORD_FORMAT)
            elif self.burst_average_data_record_version == BurstAverageDataRecordVersion.VERSION3:
                raw_data_record = self._read_data(f, self.BURST_AVERAGE_VERSION3_DATA_RECORD_FORMAT)
            else:
                raise ValueError("invalid burst/average data record version")
        elif self.id == 0x1c: # echosounder
            # echosounder is only supported by burst/average v3
            raw_data_record = self._read_data(f, self.BURST_AVERAGE_VERSION3_DATA_RECORD_FORMAT)
        elif self.id in (0x17, 0x1b): # bottom track
            raw_data_record = self._read_data(f, self.BOTTOM_TRACK_DATA_RECORD_FORMAT)
        elif self.id == 0xa0: # string data
            raw_data_record = self._read_string_data_record(f)
            print(len(raw_data_record))
        else:
            raise ValueError("invalid id")

        print("data record checksum", self.checksum(raw_data_record), self.data_record_checksum)
        assert self.checksum(
            raw_data_record) == self.data_record_checksum, "invalid data record checksum"

    def _read_string_data_record(self, f: IO):
        self.string_id = self._read_exact(f, 1)
        self.string_data = bytes()
        last_byte = None
        while last_byte is None or last_byte != b"\x00":
            last_byte = self._read_exact(f, 1)
            self.string_data += last_byte
        print(self.string_data)
        return self.string_id + self.string_data

    def _read_data(self, f: IO, data_format: List[type_field]) -> bytes:
        raw_bytes = bytes()
        for field_format in data_format:
            if len(field_format) == 3:
                field_name, field_size_bytes, field_signed = field_format
            elif len(field_format) == 4:
                field_name, field_size_bytes, field_signed, predicate = field_format
                if not predicate(self):
                    continue
            else:
                # unreachable
                raise RuntimeError
            if callable(field_size_bytes):
                field_size_bytes = field_size_bytes(self)

            raw_field = self._read_exact(f, field_size_bytes)
            raw_bytes += raw_field
            if field_signed is None:
                field_value = raw_bytes
            else:
                field_value = int.from_bytes(
                    raw_field, byteorder="little", signed=field_signed)

            setattr(self, field_name, field_value)
            self._postprocess(field_name)

        return raw_bytes

    def _postprocess(self, field_name):
        if field_name == "configuration":
            self.pressure_sensor_valid = self.configuration & 0b1000_0000_0000_0000
            self.temperature_sensor_valid = self.configuration & 0b0100_0000_0000_0000
            self.compass_sensor_valid = self.configuration & 0b0010_0000_0000_0000
            self.tilt_sensor_valid = self.configuration & 0b0001_0000_0000_0000
            self.velocity_data_included = self.configuration & 0b0000_0100_0000_0000
            self.amplitude_data_included = self.configuration & 0b0000_0010_0000_0000
            self.correlation_data_included = self.configuration & 0b0000_0001_0000_0000
        elif field_name == "num_beams_and_coordinate_system_and_num_cells":
            self.num_cells = self._reverse_bits(
                self.num_beams_and_coordinate_system_and_num_cells >> 6, width=10)
            self.num_beams = self._reverse_bits(
                self.num_beams_and_coordinate_system_and_num_cells & 0b1111, width=4)

    @staticmethod
    def _reverse_bits(n: int, width: int) -> int:
        b = "{:0{width}b}".format(n, width=width)
        return int(b[:: -1], 2)

    @staticmethod
    def checksum(data: bytes) -> int:
        """
        The Checksum is defined as a 16-bits unsigned sum of the data (16 bits). The sum shall be
        initialized to the value of 0xB58C before the checksum is calculated.
        """
        checksum = 0xb58c
        for i in range(0, len(data), 2):
            checksum += int.from_bytes(data[i: i + 2], byteorder="little")
            checksum %= 2 ** 16
        if len(data) % 2 == 1:
            checksum += data[-1] << 8
            checksum %= 2 ** 16
        return checksum

    HEADER_FORMAT: List[type_field] = [
        ("sync", 1, False),
        ("header_size", 1, False),
        ("id", 1, False),
        ("family", 1, False),
        ("data_size", 2, False),
        ("data_record_checksum", 2, False),
        ("header_checksum", 2, False),
    ]
    BURST_AVERAGE_VERSION2_DATA_RECORD_FORMAT: List[type_field] = [
        ("version", 1, False),
        ("offset_of_data", 1, False),
        ("serial_number", 4, True),
        ("configuration", 2, False),
        ("year", 1, False),
        ("month", 1, False),
        ("day", 1, False),
        ("hour", 1, False),
        ("minute", 1, False),
        ("seconds", 1, False),
        ("microsec100", 2, False),
        ("speed_of_sound", 2, False),
        ("temperature", 2, True),
        ("pressure", 4, True),
        ("heading", 2, False),
        ("pitch", 2, True),
        ("roll", 2, True),
        ("error", 2, False),
        ("status", 2, False),
        ("num_beams_and_coordinate_system_and_num_cells", 2, False),
        ("cell_size", 2, False),
        ("blanking", 2, False),
        ("velocity_range", 2, False),
        ("battery_voltage", 2, False),
        ("magnetometer_raw_x_axis", 2, True),
        ("magnetometer_raw_y_axis", 2, True),
        ("magnetometer_raw_z_axis", 2, True),
        ("accelerometer_raw_x_axis", 2, True),
        ("accelerometer_raw_y_axis", 2, True),
        ("accelerometer_raw_z_axis", 2, True),
        ("ambiguity_velocity", 2, False),
        ("dataset_description", 2, False),
        ("transmit_energy", 2, False),
        ("velocity_scaling", 1, False),
        ("power_level", 1, False),
        (None, 4, False),
        (
            "velocity_data",
            lambda self: self.num_beams * self.num_cells * 2,
            None,
            lambda self: self.velocity_data_included
        ),
        (
            "amplitude_data",
            lambda self: self.num_beams * self.num_cells * 1,
            None,
            lambda self: self.amplitude_data_included
        ),
        (
            "correlation_data",
            lambda self: self.num_beams * self.num_cells * 1,
            None,
            lambda self: self.correlation_data_included
        )
    ]
    BURST_AVERAGE_VERSION3_DATA_RECORD_FORMAT = [

    ]
    BOTTOM_TRACK_DATA_RECORD_FORMAT = [

    ]


if __name__ == "__main__":
    with open("cp0028b.ad2cp", "rb") as f:
        reader = Ad2cpReader(f, BurstAverageDataRecordVersion.VERSION2)
