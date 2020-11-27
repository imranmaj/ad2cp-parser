# Example usage (run from command line):
# python parse.py input.ad2cp


# https://support.nortekgroup.com/hc/en-us/articles/360029513952-Integrators-Guide-Signature

from typing import BinaryIO, Tuple, Union, Callable, List, Optional
from enum import Enum, unique, auto
import struct


@unique
class BurstAverageDataRecordVersion(Enum):
    VERSION2 = auto()  # Burst/Average Data Record Definition (DF2)
    VERSION3 = auto()  # Burst/Average Data Record Definition (DF3)

@unique
class DataRecordType(Enum):
    BURST_AVERAGE_VERSION2 = auto()
    BURST_AVERAGE_VERSION3 = auto()
    BOTTOM_TRACK = auto()
    STRING = auto()

@unique
class DataType(Enum):
    SIGNED_INTEGER = auto()
    UNSIGNED_INTEGER = auto()
    RAW_BYTES = auto()
    FLOAT = auto()


SIGNED_INTEGER = DataType.SIGNED_INTEGER
UNSIGNED_INTEGER = DataType.UNSIGNED_INTEGER
RAW_BYTES = DataType.RAW_BYTES
FLOAT = DataType.FLOAT

# name of the field
type_field_name = Optional[str]
# size of the field in bytes. Can be a predicate
type_field_size_bytes = Union[int, Callable[["Ad2cpDataPacket"], int]]
# whether the field is signed. If None, it is kept as bytes instead of int
type_field_data_type = DataType
# predicate to determine whether the field should exist at all
type_predicate = Callable[["Ad2cpDataPacket"], bool]
type_without_predicate_field = Tuple[type_field_name,
                                     type_field_size_bytes, type_field_data_type]
type_with_predicate_field = Tuple[type_field_name,
                                  type_field_size_bytes, type_field_data_type, type_predicate]
type_field = Union[type_without_predicate_field, type_with_predicate_field]


class NoMorePackets(Exception):
    pass


class Ad2cpReader:
    def __init__(self, f: BinaryIO, data_record_format_type: BurstAverageDataRecordVersion = BurstAverageDataRecordVersion.VERSION3, number_of_altimiter_samples: int = 0):
        self.packets = []
        counter = 0
        while True:
            try:
                self.packets.append(Ad2cpDataPacket(
                    f, data_record_format_type, number_of_altimiter_samples))
            except NoMorePackets:
                break
            else:
                counter += 1
                print(f"finished reading packet #{counter}", end="\n\n")
        print(f"successfully found and read {len(self.packets)} packets")


class Ad2cpDataPacket:
    def __init__(self, f: BinaryIO, burst_average_data_record_version: BurstAverageDataRecordVersion, number_of_altimiter_samples: int):
        self.burst_average_data_record_version = burst_average_data_record_version
        self.number_of_altimiter_samples = number_of_altimiter_samples
        self.data_record_type: Optional[DataRecordType] = None
        self._read_header(f)
        self._read_data_record(f)

    @staticmethod
    def _read_exact(f: BinaryIO, total_num_bytes_to_read: int) -> bytes:
        """
        Drives a stream until an exact amount of bytes is read from it.
        This is necessary because a single read may not return the correct number of bytes.
        """

        all_bytes_read = bytes()
        if total_num_bytes_to_read <= 0:
            return all_bytes_read
        last_bytes_read = None
        while last_bytes_read is None or (len(last_bytes_read) > 0 and len(all_bytes_read) < total_num_bytes_to_read):
            last_bytes_read = f.read(total_num_bytes_to_read - len(all_bytes_read))
            if len(last_bytes_read) == 0:
                raise NoMorePackets
            else:
                all_bytes_read += last_bytes_read
        return all_bytes_read

    def _read_header(self, f: BinaryIO):
        """
        Reads the header part of the AD2CP packet from the stream
        """

        raw_header = self._read_data(f, self.HEADER_FORMAT)
        print("header checksum      calculated:", self.checksum(
            raw_header[: -2]), "expected:", self.header_checksum)
        # don't include the last 2 bytes, which is the header checksum itself
        assert self.checksum(
            raw_header[: -2]) == self.header_checksum, "invalid header checksum"

    def _read_data_record(self, f: BinaryIO):
        """
        Reads the data record part of the AD2CP packet from the stream
        """

        print("data record type id:", self.id)
        # TODO: figure out where to send the other ids
        if self.id in (0x15, 0x16, 0x18, ):  # burst/average
            if self.burst_average_data_record_version == BurstAverageDataRecordVersion.VERSION2:
                data_record_format = self.BURST_AVERAGE_VERSION2_DATA_RECORD_FORMAT
                self.data_record_type = DataRecordType.BURST_AVERAGE_VERSION2
            elif self.burst_average_data_record_version == BurstAverageDataRecordVersion.VERSION3:
                data_record_format = self.BURST_AVERAGE_VERSION3_DATA_RECORD_FORMAT
                self.data_record_type = DataRecordType.BURST_AVERAGE_VERSION3
            else:
                raise ValueError("invalid burst/average data record version")
        elif self.id == 0x1c:  # echosounder
            # echosounder is only supported by burst/average v3
            data_record_format = self.BURST_AVERAGE_VERSION3_DATA_RECORD_FORMAT
            self.data_record_type = DataRecordType.BURST_AVERAGE_VERSION3
        elif self.id in (0x17, 0x1b):  # bottom track
            data_record_format = self.BOTTOM_TRACK_DATA_RECORD_FORMAT
            self.data_record_type = DataRecordType.BOTTOM_TRACK
        elif self.id == 0xa0:  # string data
            data_record_format = self.STRING_DATA_RECORD_FORMAT
            self.data_record_type = DataRecordType.STRING
        else:
            raise ValueError("invalid data record type id")

        raw_data_record = self._read_data(f, data_record_format)
        print("data record checksum calculated:", self.checksum(
            raw_data_record), "expected:", self.data_record_checksum)
        assert self.checksum(
            raw_data_record) == self.data_record_checksum, "invalid data record checksum"

    def _read_data(self, f: BinaryIO, data_format: List[type_field]) -> bytes:
        """
        Reads data from the stream, interpreting the data using the given format
        """

        raw_bytes = bytes()  # combination of all raw fields
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
            # all numbers are little endian
            if field_signed == DataType.RAW_BYTES:
                field_value = raw_field
            elif field_signed == DataType.SIGNED_INTEGER:
                field_value = int.from_bytes(
                    raw_field, byteorder="little", signed=True)
            elif field_signed == DataType.UNSIGNED_INTEGER:
                field_value = int.from_bytes(
                    raw_field, byteorder="little", signed=False)
            elif field_signed == DataType.FLOAT and field_size_bytes == 4:
                field_value = struct.unpack("f", raw_field)
            elif field_signed == DataType.FLOAT and field_size_bytes == 8:
                field_value = struct.unpack("d", raw_field)
            if field_name is not None:
                setattr(self, field_name, field_value)
                self._postprocess(field_name)

        return raw_bytes

    def _postprocess(self, field_name):
        """
        Calculates values based on parsed data. This should be called immediately after
        parsing each field in a data record.
        """

        if self.data_record_type == DataRecordType.BURST_AVERAGE_VERSION2:
            if field_name == "configuration":
                self.pressure_sensor_valid = self.configuration & 0b0000_0000_0000_0001 > 0
                self.temperature_sensor_valid = self.configuration & 0b0000_0000_0000_0010 > 0
                self.compass_sensor_valid = self.configuration & 0b0000_0000_0000_0100 > 0
                self.tilt_sensor_valid = self.configuration & 0b0000_0000_0000_1000 > 0
                self.velocity_data_included = self.configuration & 0b0000_0000_0010_0000 > 0
                self.amplitude_data_included = self.configuration & 0b0000_0000_0100_0000 > 0
                self.correlation_data_included = self.configuration & 0b0000_0000_1000_0000 > 0
            elif field_name == "num_beams_and_coordinate_system_and_num_cells":
                self.num_cells = self.num_beams_and_coordinate_system_and_num_cells & 0b0000_0011_1111_1111
                self.coordinate_system = (
                    self.num_beams_and_coordinate_system_and_num_cells & 0b0000_1100_0000_0000) >> 10
                self.num_beams = (
                    self.num_beams_and_coordinate_system_and_num_cells & 0b1111_0000_0000_0000) >> 12
        elif self.data_record_type == DataRecordType.BURST_AVERAGE_VERSION3:
            if field_name == "configuration":
                self.pressure_sensor_valid = self.configuration & 0b0000_0000_0000_0001 > 0
                self.temperature_sensor_valid = self.configuration & 0b0000_0000_0000_0010 > 0
                self.compass_sensor_valid = self.configuration & 0b0000_0000_0000_0100 > 0
                self.tilt_sensor_valid = self.configuration & 0b0000_0000_0000_1000 > 0
                self.velocity_data_included = self.configuration & 0b0000_0000_0010_0000 > 0
                self.amplitude_data_included = self.configuration & 0b0000_0000_0100_0000 > 0
                self.correlation_data_included = self.configuration & 0b0000_0000_1000_0000 > 0
                self.altimiter_data_included = self.configuration & 0b0000_0001_0000_0000 > 0
                self.altimiter_raw_data_included = self.configuration & 0b0000_0010_0000_0000 > 0
                self.ast_data_included = self.configuration & 0b0000_0100_0000_0000 > 0
                self.echo_sounder_data_included = self.configuration & 0b0000_1000_0000_0000 > 0
                self.ahrs_data_included = self.configuration & 0b0001_0000_0000_0000 > 0
                self.percentage_good_data_included = self.configuration & 0b0010_0000_0000_0000 > 0
                self.std_dev_data_included = self.configuration & 0b0100_0000_0000_0000 > 0
            elif field_name == "num_beams_and_coordinate_system_and_num_cells":
                if self.echo_sounder_data_included:
                    self.num_echo_sounder_cells = self.num_beams_and_coordinate_system_and_num_cells
                else:
                    self.num_cells = self.num_beams_and_coordinate_system_and_num_cells & 0b0000_0011_1111_1111
                    self.coordinate_system = (
                        self.num_beams_and_coordinate_system_and_num_cells & 0b0000_1100_0000_0000) >> 10
                    self.num_beams = (
                        self.num_beams_and_coordinate_system_and_num_cells & 0b1111_0000_0000_0000) >> 12
            elif field_name == "ambiguity_velocity_or_echo_sounder_frequency":
                if self.echo_sounder_data_included:
                    # TODO: this is listed in the table as "echo sounder frequency", but the description
                    # says "number of echo sounder cells". It is probably the frequency and not the number of cells
                    #  because the number of cells is already listed under "num_beams_and_coordinate_system_and_num_cells"
                    self.echo_sounder_frequency = self.ambiguity_velocity_or_echo_sounder_frequency
                else:
                    self.ambiguity_velocity = self.ambiguity_velocity_or_echo_sounder_frequency
        elif self.data_record_type == DataRecordType.BOTTOM_TRACK:
            if field_name == "configuration":
                self.pressure_sensor_valid = self.configuration & 0b0000_0000_0000_0001 > 0
                self.temperature_sensor_valid = self.configuration & 0b0000_0000_0000_0010 > 0
                self.compass_sensor_valid = self.configuration & 0b0000_0000_0000_0100 > 0
                self.tilt_sensor_valid = self.configuration & 0b0000_0000_0000_1000 > 0
                self.velocity_data_included = self.configuration & 0b0000_0000_0010_0000 > 0
                self.distance_data_included = self.configuration & 0b0000_0001_0000_0000 > 0
                self.figure_of_merit_data_included = self.configuration & 0b0000_0010_0000_0000 > 0
            elif field_name == "num_beams_and_coordinate_system_and_num_cells":
                self.num_cells = self.num_beams_and_coordinate_system_and_num_cells & 0b0000_0011_1111_1111
                self.coordinate_system = (
                    self.num_beams_and_coordinate_system_and_num_cells & 0b0000_1100_0000_0000) >> 10
                self.num_beams = (
                    self.num_beams_and_coordinate_system_and_num_cells & 0b1111_0000_0000_0000) >> 12

    @staticmethod
    def checksum(data: bytes) -> int:
        """
        Computes the checksum for the given data
        """

        checksum = 0xb58c
        for i in range(0, len(data), 2):
            checksum += int.from_bytes(data[i: i + 2], byteorder="little")
            checksum %= 2 ** 16
        if len(data) % 2 == 1:
            checksum += data[-1] << 8
            checksum %= 2 ** 16
        return checksum

    # data formats (passed to self._read_data)
    HEADER_FORMAT: List[type_field] = [
        ("sync", 1, UNSIGNED_INTEGER),
        ("header_size", 1, UNSIGNED_INTEGER),
        ("id", 1, UNSIGNED_INTEGER),
        ("family", 1, UNSIGNED_INTEGER),
        ("data_record_size", 2, UNSIGNED_INTEGER),
        ("data_record_checksum", 2, UNSIGNED_INTEGER),
        ("header_checksum", 2, UNSIGNED_INTEGER),
    ]
    STRING_DATA_RECORD_FORMAT: List[type_field] = [
        ("string_data_id", 1, RAW_BYTES),
        ("string_data", lambda self: self.data_record_size - 1, RAW_BYTES)
    ]
    BURST_AVERAGE_VERSION2_DATA_RECORD_FORMAT: List[type_field] = [
        ("version", 1, UNSIGNED_INTEGER),
        ("offset_of_data", 1, UNSIGNED_INTEGER),
        ("serial_number", 4, UNSIGNED_INTEGER),
        ("configuration", 2, UNSIGNED_INTEGER),
        ("year", 1, UNSIGNED_INTEGER),
        ("month", 1, UNSIGNED_INTEGER),
        ("day", 1, UNSIGNED_INTEGER),
        ("hour", 1, UNSIGNED_INTEGER),
        ("minute", 1, UNSIGNED_INTEGER),
        ("seconds", 1, UNSIGNED_INTEGER),
        ("microsec100", 2, UNSIGNED_INTEGER),
        ("speed_of_sound", 2, UNSIGNED_INTEGER),
        ("temperature", 2, SIGNED_INTEGER),
        ("pressure", 4, UNSIGNED_INTEGER),
        ("heading", 2, UNSIGNED_INTEGER),
        ("pitch", 2, SIGNED_INTEGER),
        ("roll", 2, SIGNED_INTEGER),
        ("error", 2, UNSIGNED_INTEGER),
        ("status", 2, UNSIGNED_INTEGER),
        ("num_beams_and_coordinate_system_and_num_cells", 2, UNSIGNED_INTEGER),
        ("cell_size", 2, UNSIGNED_INTEGER),
        ("blanking", 2, UNSIGNED_INTEGER),
        ("velocity_range", 2, UNSIGNED_INTEGER),
        ("battery_voltage", 2, UNSIGNED_INTEGER),
        ("magnetometer_raw_x_axis", 2, SIGNED_INTEGER),
        ("magnetometer_raw_y_axis", 2, SIGNED_INTEGER),
        ("magnetometer_raw_z_axis", 2, SIGNED_INTEGER),
        ("accelerometer_raw_x_axis", 2, SIGNED_INTEGER),
        ("accelerometer_raw_y_axis", 2, SIGNED_INTEGER),
        ("accelerometer_raw_z_axis", 2, SIGNED_INTEGER),
        ("ambiguity_velocity", 2, UNSIGNED_INTEGER),
        ("dataset_description", 2, UNSIGNED_INTEGER),
        ("transmit_energy", 2, UNSIGNED_INTEGER),
        ("velocity_scaling", 1, SIGNED_INTEGER),
        ("power_level", 1, SIGNED_INTEGER),
        (None, 4, UNSIGNED_INTEGER),
        # TODO: this data is interpreted as raw bytes instead of integers
        # because it actually contains a large series of integers. The raw bytes
        # need to be split into integers during postprocessing
        (
            "velocity_data",
            lambda self: self.num_beams * self.num_cells * 2,
            RAW_BYTES,
            lambda self: self.velocity_data_included
        ),
        (
            "amplitude_data",
            lambda self: self.num_beams * self.num_cells * 1,
            RAW_BYTES,
            lambda self: self.amplitude_data_included
        ),
        (
            "correlation_data",
            lambda self: self.num_beams * self.num_cells * 1,
            RAW_BYTES,
            lambda self: self.correlation_data_included
        )
    ]
    BURST_AVERAGE_VERSION3_DATA_RECORD_FORMAT = [
        ("version", 1, UNSIGNED_INTEGER),
        ("offset_of_data", 1, UNSIGNED_INTEGER),
        ("configuration", 2, UNSIGNED_INTEGER),
        ("serial_number", 4, UNSIGNED_INTEGER),
        ("year", 1, UNSIGNED_INTEGER),
        ("month", 1, UNSIGNED_INTEGER),
        ("day", 1, UNSIGNED_INTEGER),
        ("hour", 1, UNSIGNED_INTEGER),
        ("minute", 1, UNSIGNED_INTEGER),
        ("seconds", 1, UNSIGNED_INTEGER),
        ("microsec100", 2, UNSIGNED_INTEGER),
        ("speed_of_sound", 2, UNSIGNED_INTEGER),
        ("temperature", 2, SIGNED_INTEGER),
        ("pressure", 4, UNSIGNED_INTEGER),
        ("heading", 2, UNSIGNED_INTEGER),
        ("pitch", 2, SIGNED_INTEGER),
        ("roll", 2, SIGNED_INTEGER),
        ("num_beams_and_coordinate_system_and_num_cells", 2, UNSIGNED_INTEGER),
        ("cell_size", 2, UNSIGNED_INTEGER),
        ("blanking", 2, UNSIGNED_INTEGER),
        ("nominal_correlation", 1, UNSIGNED_INTEGER),
        ("temperature_from_pressure_sensor", 1, UNSIGNED_INTEGER),
        ("battery_voltage", 2, UNSIGNED_INTEGER),
        ("magnetometer_raw_x_axis", 2, SIGNED_INTEGER),
        ("magnetometer_raw_y_axis", 2, SIGNED_INTEGER),
        ("magnetometer_raw_z_axis", 2, SIGNED_INTEGER),
        ("accelerometer_raw_x_axis", 2, SIGNED_INTEGER),
        ("accelerometer_raw_y_axis", 2, SIGNED_INTEGER),
        ("accelerometer_raw_z_axis", 2, SIGNED_INTEGER),
        ("ambiguity_velocity_or_echo_sounder_frequency", 2, UNSIGNED_INTEGER),
        ("dataset_description", 2, UNSIGNED_INTEGER),
        ("transmit_energy", 2, UNSIGNED_INTEGER),
        ("velocity_scaling", 1, SIGNED_INTEGER),
        ("power_level", 1, SIGNED_INTEGER),
        ("magnetometer_temperature", 2, SIGNED_INTEGER),
        ("real_time_clock_temperature", 2, SIGNED_INTEGER),
        ("error", 2, UNSIGNED_INTEGER),
        ("status0", 2, UNSIGNED_INTEGER),
        ("status", 4, UNSIGNED_INTEGER),
        ("ensemble_counter", 4, UNSIGNED_INTEGER),
        (
            "velocity_data",
            lambda self: self.num_beams * self.num_cells * 2,
            RAW_BYTES,
            lambda self: self.velocity_data_included
        ),
        (
            "amplitude_data",
            lambda self: self.num_beams * self.num_cells * 1,
            RAW_BYTES,
            lambda self: self.amplitude_data_included
        ),
        (
            "correlation_data",
            lambda self: self.num_beams * self.num_cells * 1,
            RAW_BYTES,
            lambda self: self.correlation_data_included
        ),
        ("altimiter_distance", 4, FLOAT, lambda self: self.altimiter_data_included),
        ("altimiter_quality", 2, UNSIGNED_INTEGER,
         lambda self: self.altimiter_data_included),
        ("ast_distance", 4, FLOAT, lambda self: self.ast_data_included),
        ("ast_quality", 2, UNSIGNED_INTEGER, lambda self: self.ast_data_included),
        ("ast_offset_10us", 2, SIGNED_INTEGER,
         lambda self: self.ast_data_included),
        ("ast_pressure", 4, FLOAT, lambda self: self.ast_data_included),
        ("altimiter_spare", 8, RAW_BYTES, lambda self: self.ast_data_included),
        (
            "altimiter_raw_data_num_samples",
            # TODO: other counts, like the number of beams or number of cells, can be found
            # by parsing the data itself, but it seems like the number of altimiter samples
            # must be known beforehand. Is there a way to calculate this instead?
            lambda self: self.number_of_altimiter_samples * 2,
            RAW_BYTES,
            lambda self: self.altimiter_raw_data_included
        ),
        ("altimiter_raw_data_sample_distance", 2, UNSIGNED_INTEGER,
         lambda self: self.altimiter_raw_data_included),
        ("altimiter_raw_data_samples", lambda self: 2, RAW_BYTES,
         lambda self: self.altimiter_raw_data_included),
        (
            "echo_sounder_data",
            lambda self: self.num_cells * 2,
            RAW_BYTES,
            lambda self: self.echo_sounder_data_included
        ),
        ("ahrs_rotation_matrix_m11", 4, FLOAT,
         lambda self: self.ahrs_data_included),
        ("ahrs_rotation_matrix_m12", 4, FLOAT,
         lambda self: self.ahrs_data_included),
        ("ahrs_rotation_matrix_m13", 4, FLOAT,
         lambda self: self.ahrs_data_included),
        ("ahrs_rotation_matrix_m21", 4, FLOAT,
         lambda self: self.ahrs_data_included),
        ("ahrs_rotation_matrix_m22", 4, FLOAT,
         lambda self: self.ahrs_data_included),
        ("ahrs_rotation_matrix_m23", 4, FLOAT,
         lambda self: self.ahrs_data_included),
        ("ahrs_rotation_matrix_m31", 4, FLOAT,
         lambda self: self.ahrs_data_included),
        ("ahrs_rotation_matrix_m32", 4, FLOAT,
         lambda self: self.ahrs_data_included),
        ("ahrs_rotation_matrix_m33", 4, FLOAT,
         lambda self: self.ahrs_data_included),
        ("ahrs_quaternions_w", 4, FLOAT, lambda self: self.ahrs_data_included),
        ("ahrs_quaternions_x", 4, FLOAT, lambda self: self.ahrs_data_included),
        ("ahrs_quaternions_y", 4, FLOAT, lambda self: self.ahrs_data_included),
        ("ahrs_quaternions_z", 4, FLOAT, lambda self: self.ahrs_data_included),
        ("ahrs_gyro_x", 4, FLOAT, lambda self: self.ahrs_data_included),
        ("ahrs_gyro_y", 4, FLOAT, lambda self: self.ahrs_data_included),
        ("ahrs_gyro_z", 4, FLOAT, lambda self: self.ahrs_data_included),
        (
            "percentage_good_data",
            lambda self: self.num_cells,
            UNSIGNED_INTEGER,
            lambda self: self.percentage_good_data_included
        ),
        # only the pitch field is labeled as included when the "std dev data included"
        # bit is set, but this is likely a mistake
        ("std_dev_pitch", 2, SIGNED_INTEGER,
         lambda self: self.std_dev_data_included),
        ("std_dev_roll", 2, SIGNED_INTEGER,
         lambda self: self.std_dev_data_included),
        ("std_dev_heading", 2, SIGNED_INTEGER,
         lambda self: self.std_dev_data_included),
        ("std_dev_pressure", 2, SIGNED_INTEGER,
         lambda self: self.std_dev_data_included),
        (None, 24, RAW_BYTES, lambda self: self.std_dev_data_included)
    ]
    BOTTOM_TRACK_DATA_RECORD_FORMAT = [
        ("version", 1, UNSIGNED_INTEGER),
        ("offset_of_data", 1, UNSIGNED_INTEGER),
        ("configuration", 2, UNSIGNED_INTEGER),
        ("serial_number", 4, UNSIGNED_INTEGER),
        ("year", 1, UNSIGNED_INTEGER),
        ("month", 1, UNSIGNED_INTEGER),
        ("day", 1, UNSIGNED_INTEGER),
        ("hour", 1, UNSIGNED_INTEGER),
        ("minute", 1, UNSIGNED_INTEGER),
        ("seconds", 1, UNSIGNED_INTEGER),
        ("microsec100", 2, UNSIGNED_INTEGER),
        ("speed_of_sound", 2, UNSIGNED_INTEGER),
        ("temperature", 2, SIGNED_INTEGER),
        ("pressure", 4, UNSIGNED_INTEGER),
        ("heading", 2, UNSIGNED_INTEGER),
        ("pitch", 2, SIGNED_INTEGER),
        ("roll", 2, SIGNED_INTEGER),
        ("num_beams_and_coordinate_system_and_num_cells", 2, UNSIGNED_INTEGER),
        ("cell_size", 2, UNSIGNED_INTEGER),
        ("blanking", 2, UNSIGNED_INTEGER),
        ("nominal_correlation", 1, UNSIGNED_INTEGER),
        (None, 1, RAW_BYTES),
        ("battery_voltage", 2, UNSIGNED_INTEGER),
        ("magnetometer_raw_x_axis", 2, SIGNED_INTEGER),
        ("magnetometer_raw_y_axis", 2, SIGNED_INTEGER),
        ("magnetometer_raw_z_axis", 2, SIGNED_INTEGER),
        ("accelerometer_raw_x_axis", 2, SIGNED_INTEGER),
        ("accelerometer_raw_y_axis", 2, SIGNED_INTEGER),
        ("accelerometer_raw_z_axis", 2, SIGNED_INTEGER),
        ("ambiguity_velocity", 4, UNSIGNED_INTEGER),
        ("dataset_description", 2, UNSIGNED_INTEGER),
        ("transmit_energy", 2, UNSIGNED_INTEGER),
        ("velocity_scaling", 1, SIGNED_INTEGER),
        ("power_level", 1, SIGNED_INTEGER),
        ("magnetometer_temperature", 2, SIGNED_INTEGER),
        ("real_time_clock_temperature", 2, SIGNED_INTEGER),
        ("error", 4, UNSIGNED_INTEGER),
        ("status", 4, UNSIGNED_INTEGER),
        ("ensemble_counter", 4, UNSIGNED_INTEGER),
        (
            "velocity_data",
            lambda self: self.num_beams * 4,
            RAW_BYTES,
            lambda self: self.velocity_data_included
        ),
        (
            "distance_data",
            lambda self: self.num_beams * 4,
            RAW_BYTES,
            lambda self: self.distance_data_included
        ),
        (
            "figure_of_merit_data",
            lambda self: self.num_beams * 2,
            RAW_BYTES,
            lambda self: self.figure_of_merit_data_included
        )
    ]


if __name__ == "__main__":
    import sys
    with open(sys.argv[1], "rb") as f:
        reader = Ad2cpReader(f)

    # for packet in reader.packets:
    #     print(packet.__dict__)