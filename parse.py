# Example usage (run from command line):
# python parse.py input.ad2cp


# https://support.nortekgroup.com/hc/en-us/articles/360029513952-Integrators-Guide-Signature

from typing import BinaryIO, Tuple, Union, Callable, List, Dict, Optional, Any
from enum import Enum, unique, auto
from datetime import datetime
import struct
import math
import os

import numpy as np
from numpy.core.numeric import NaN
import xarray as xr
import pytz


@unique
class BurstAverageDataRecordVersion(Enum):
    VERSION2 = auto()  # Burst/Average Data Record Definition (DF2)
    VERSION3 = auto()  # Burst/Average Data Record Definition (DF3)


@unique
class DataRecordType(Enum):
    BURST_VERSION2 = auto()
    BURST_VERSION3 = auto()
    AVERAGE_VERSION2 = auto()
    AVERAGE_VERSION3 = auto()
    BOTTOM_TRACK = auto()
    STRING = auto()


@unique
class DataType(Enum):
    RAW_BYTES = auto()
    STRING = auto()
    SIGNED_INTEGER = auto()
    UNSIGNED_INTEGER = auto()
    FLOAT = auto()
    # TODO: figure out what this is in binary
    SIGNED_FRACTION = auto()


RAW_BYTES = DataType.RAW_BYTES
STRING = DataType.STRING
SIGNED_INTEGER = DataType.SIGNED_INTEGER
UNSIGNED_INTEGER = DataType.UNSIGNED_INTEGER
FLOAT = DataType.FLOAT
SIGNED_FRACTION = DataType.SIGNED_FRACTION


@unique
class Dimension(Enum):
    TIME = "time"
    BEAM = "beam"
    RANGE_BIN = "range_bin"
    NUM_ALTIMETER_SAMPLES = "num_altimeter_samples"


class Field:
    def __init__(self, field_name: Optional[str], field_entry_size_bytes: Union[int, Callable[["Ad2cpDataPacket"], int]], field_entry_data_type: DataType, *, field_shape: Union[List[int], Callable[["Ad2cpDataPacket"], List[int]]] = [], field_exists_predicate: Callable[["Ad2cpDataPacket"], bool] = lambda _: True):
        """
        field_name: Name of the field. If None, the field is parsed but ignored
        field_entry_size_bytes: Size of each entry within the field, in bytes. 
            In most cases, the entry is the field itself, but sometimes the field
            contains a list of entries.
        field_entry_data_type: Data type of each entry in the field
        field_shape: Shape of entries within the field. 
            [] (the default) means the entry is the field itself,
            [n] means the field consists of a list of n entries,
            [n, m] means the field consists of a two dimensional array with
                n number of m length arrays,
            etc.
        field_exists_predicate: Tests to see whether the field should be parsed at all
        """

        self.field_name = field_name
        self.field_entry_size_bytes = field_entry_size_bytes
        self.field_entry_data_type = field_entry_data_type
        self.field_shape = field_shape
        self.field_exists_predicate = field_exists_predicate

    @staticmethod
    def dimensions(field_name: str, data_record_format: List["Field"]) -> List[Dimension]:
        # TODO: altimeter spare (but it's not included in final dataset)
        if data_record_format == Ad2cpDataPacket.BOTTOM_TRACK_DATA_RECORD_FORMAT:
            if field_name in ("velocity_data", "distance_data", "figure_of_merit_data"):
                return [Dimension.TIME, Dimension.BEAM]
        else:
            if field_name in ("velocity_data", "amplitude_data", "correlation_data"):
                return [Dimension.TIME, Dimension.BEAM, Dimension.RANGE_BIN]
            elif field_name in ("echo_sounder_data", "percentage_good_data"):
                return [Dimension.TIME, Dimension.RANGE_BIN]
            elif field_name == "altimeter_raw_data_samples":
                return [Dimension.TIME, Dimension.NUM_ALTIMETER_SAMPLES]
        return [Dimension.TIME]


F = Field  # use F instead of Field to make the repeated fields easier to read

ECHOPYPE_VERSION = 0


class NoMorePackets(Exception):
    pass


class SetGroupsAd2cp:
    def __init__(self, parser: "Ad2cpReader", output_path: str):
        self.parser = parser
        self.ds = parser.parse_raw()
        print(self.ds)
        self.output_path = output_path

    def write(self, ds: xr.Dataset, group: str):
        if os.path.exists(self.output_path):
            mode = "a"
        else:
            mode = "w"
        ds.to_netcdf(path=self.output_path, mode=mode, group=group)

    def save(self):
        self.set_environment()
        self.set_platform()
        self.set_beam()
        self.set_vendor_specific()

    def set_environment(self):
        ds = xr.Dataset(data_vars={
            "sound_speed_indicative": self.ds.get("speed_of_sound"),
            "temperature": self.ds.get("temperature"),
            "pressure": self.ds.get("pressure")
        },
            coords={
            "time_burst": self.ds.get("time_burst", []),
            "time_average": self.ds.get("time_average", [])
        })
        self.write(ds, "Environment")

    def set_platform(self):
        ds = xr.Dataset(
            data_vars={
                "heading": self.ds.get("heading"),
                "pitch": self.ds.get("pitch"),
                "roll": self.ds.get("roll"),
                "magnetometer_raw_x": self.ds.get("magnetometer_raw_x"),
                "magnetometer_raw_y": self.ds.get("magnetometer_raw_y"),
                "magnetometer_raw_z": self.ds.get("magnetometer_raw_z"),
            },
            coords={
                "time_burst": self.ds.get("time_burst"),
                "time_average": self.ds.get("time_average"),
                "beam": self.ds.get("beam"),
                "range_bin": self.ds.get("range_bin")
            },
            attrs={
                "platform_name": self.ui_param["platform_name"],
                "platform_type": self.ui_param["platform_type"],
                "platform_code_ICES": self.ui_param["platform_code_ICES"]
            }
        )
        self.write(ds, "Platform")

    def set_beam(self):
        data_vars = {
            "number_of_beams": self.ds.get("num_beams"),
            "coordinate_system": self.ds.get("coordinate_system"),
            "number_of_cells": self.ds.get("num_cells"),
            "blanking": self.ds.get("blanking"),
            "cell_size": self.ds.get("cell_size"),
            "velocity_range": self.ds.get("velocity_range"),
            # TODO: should the following 2 just be empty if they dont exist or not be included at all?
            "echosounder_frequency": self.ds.get("echo_sounder_frequency"),
            "ambiguity_velocity": self.ds.get("ambiguity_velocity"),
            "data_set_description": self.ds.get("dataset_description"),
            "transmit_energy": self.ds.get("transmit_energy"),
            "velocity_scaling": self.ds.get("velocity_scaling"),
            "velocity": self.ds.get("velocity_data"),
            "amplitude": self.ds.get("amplitude_data"),
            "correlation": self.ds.get("correlation_data"),
            "echosounder": self.ds.get("echosounder_data"),
            "figure_of_merit": self.ds.get("figure_of_merit_data"),
            "altimeter_distance": self.ds.get("altimeter_distance"),
            "altimeter_quality": self.ds.get("altimeter_quality"),
            "ast_distance": self.ds.get("ast_distance"),
            "ast_quality": self.ds.get("ast_quality"),
            "ast_offset_10us": self.ds.get("ast_offset_10us"),
            "ast_pressure": self.ds.get("ast_pressure"),
            "altimeter_spare": self.ds.get("altimeter_spare"),
            "altimeter_raw_data_num_samples": self.ds.get("altimeter_raw_data_num_samples"),
            "altimeter_raw_data_sample_distance": self.ds.get("altimeter_raw_data_sample_distance"),
            "altimeter_raw_data_samples": self.ds.get("altimeter_raw_data_samples")
        }
        # if self.ds["echo_sounder_data_included"][0]:  # echosounder
        #     data_vars["echosounder_frequency"] = self.ds.get(
        #         "echo_sounder_frequency")
        # else:
        #     data_vars["ambiguity_velocity"] = self.ds.get("ambiguity_velocity")

        ds = xr.Dataset(
            data_vars=data_vars,
            coords={
                "time_burst": self.ds.get("time_burst"),
                "time_average": self.ds.get("time_average"),
                "beam": self.ds.get("beam"),
                "range_bin": self.ds.get("range_bin"),
                "altimeter_sample_bin": self.ds.get("altimeter_sample_bin"),
                "range_bin_echosounder": self.ds.get("num_beams_and_coordinate_system_and_num_cells")
            }
        )
        self.write(ds, "Beam")

    def set_vendor_specific(self):
        attrs = {
            "offset_of_data": self.ds.get("offset_of_data"),
            "pressure_sensor_valid": self.ds.get("pressure_sensor_valid"),
            "temperature_sensor_valid": self.ds.get("temperature_sensor_valid"),
            "compass_sensor_valid": self.ds.get("compass_sensor_valid"),
            "tilt_sensor_valid": self.ds.get("tilt_sensor_valid"),
            "velocity_data_included": self.ds.get("velocity_data_included"),
            "amplitude_data_included": self.ds.get("amplitude_data_included"),
            "correlation_data_included": self.ds.get("correlation_data_included"),
            "altimeter_data_included": self.ds.get("altimeter_data_included"),
            "altimeter_raw_data_included": self.ds.get("altimeter_raw_data_included"),
            "ast_data_included": self.ds.get("ast_data_included"),
            "echo_sounder_data_included": self.ds.get("echo_sounder_data_included"),
            "ahrs_data_included": self.ds.get("ahrs_data_included"),
            "percentage_good_data_included": self.ds.get("percentage_good_data_included"),
            "std_dev_data_included": self.ds.get("std_dev_data_included"),
            "figure_of_merit_data_included": self.ds.get("figure_of_merit_data_included"),
        }
        attrs = {field_name: field_value.data[0] for field_name, field_value in attrs.items(
        ) if field_value is not None}
        ds = xr.Dataset(data_vars={
            "data_record_version": self.ds.get("version"),
            "error": self.ds.get("error"),
            "status": self.ds.get("status"),
            "status0": self.ds.get("status0"),
            "battery_voltage": self.ds.get("battery_voltage"),
            "power_level": self.ds.get("power_level"),
            "temperature_of_pressure_sensor": self.ds.get("temperature_from_pressure_sensor"),
            "nominal_correlation": self.ds.get("nominal_correlation"),
            "magnetometer_temperature": self.ds.get("magnetometer_temperature"),
            "real_time_clock_temperature": self.ds.get("real_time_clock_temperature"),
            "ensemble_counter": self.ds.get("ensemble_counter"),
            "ahrs_rotation_matrix_mij": ("mij", [
                self.ds.get("ahrs_rotation_matrix_m11"),
                self.ds.get("ahrs_rotation_matrix_m12"),
                self.ds.get("ahrs_rotation_matrix_m13"),
                self.ds.get("ahrs_rotation_matrix_m21"),
                self.ds.get("ahrs_rotation_matrix_m22"),
                self.ds.get("ahrs_rotation_matrix_m23"),
                self.ds.get("ahrs_rotation_matrix_m31"),
                self.ds.get("ahrs_rotation_matrix_m32"),
                self.ds.get("ahrs_rotation_matrix_m33")
            ]),
            "ahrs_quaternions_wxyz": ("wxyz", [
                self.ds.get("ahrs_quaternions_w"),
                self.ds.get("ahrs_quaternions_x"),
                self.ds.get("ahrs_quaternions_y"),
                self.ds.get("ahrs_quaternions_z"),
            ]),
            "ahrs_gyro_xyz": ("xyz", [
                self.ds.get("ahrs_gyro_x"),
                self.ds.get("ahrs_gyro_y"),
                self.ds.get("ahrs_gyro_z"),
            ]),
            "percentage_good_data": self.ds.get("percentage_good_data"),
            "std_dev_pitch": self.ds.get("std_dev_pitch"),
            "std_dev_roll": self.ds.get("std_dev_roll"),
            "std_dev_heading": self.ds.get("std_dev_heading"),
            "std_dev_pressure": self.ds.get("std_dev_pressure"),
        },
            coords={
            "time_burst": self.ds.get("time_burst"),
            "time_average": self.ds.get("time_average"),
            "beam": self.ds.get("beam"),
            "range_bin": self.ds.get("range_bin"),
        }, attrs=attrs)
        ds = ds.reindex({
            "mij": ["11", "12", "13", "21", "22", "23", "31", "32", "33"],
            "wxyz": ["w", "x", "y", "z"],
            "xyz": ["x", "y", "z"],
        })
        self.write(ds, "Vendor")


class Ad2cpReader:
    def __init__(self, f: BinaryIO, burst_average_data_record_version: BurstAverageDataRecordVersion = BurstAverageDataRecordVersion.VERSION3):
        self.packets = []
        while True:
            try:
                self.packets.append(Ad2cpDataPacket(
                    f, burst_average_data_record_version))
            except NoMorePackets:
                break
        print(f"successfully found and read {len(self.packets)} packets")

    def parse_raw(self) -> xr.Dataset:
        ds = xr.Dataset(attrs={"string_data": dict()})

        max_beam_count = max(
            self.packets, key=lambda p: p.data["num_beams"] if "num_beams" in p.data else 0).data["num_beams"]
        max_range_bin_count = max(
            self.packets, key=lambda p: p.data["num_cells"] if "num_cells" in p.data else 0).data["num_cells"]
        # TODO: fill packets with missing beams with Nones up to the max beam count
        for packet in self.packets:
            for field_name, field_value in packet.data.items():
                if isinstance(field_value, np.ndarray):
                    if len(field_value[0]) < max_range_bin_count:
                        field_value = [np.concatenate(
                            (beam, [np.nan] * (max_range_bin_count - len(field_value[0])))) for beam in field_value]
                        packet.data[field_name] = np.array(field_value)
                    if len(field_value) < max_beam_count:
                        field_value = np.concatenate(
                            (field_value, [[np.nan] * max_range_bin_count for _ in range(max_beam_count - len(field_value))]))
                        packet.data[field_name] = np.array(field_value)

        for packet in self.packets:
            if packet.data_record_type == DataRecordType.STRING:
                ds.attrs["string_data"][packet.data["string_data_id"]
                                        ] = packet.data["string_data"]
            else:
                if packet.data_record_type in (DataRecordType.BURST_VERSION2, DataRecordType.BURST_VERSION3):
                    time = "time_burst"
                else:
                    time = "time_average"
                data_vars = dict()
                for field_name, field_value in packet.data.items():
                    # TODO might not work with altimeter_spare
                    data_vars[field_name] = (tuple(dim.value for dim in Field.dimensions(
                        field_name, packet.data_record_format)), [field_value])
                ds = xr.merge([
                    ds,
                    xr.Dataset(
                        data_vars=data_vars,
                        coords={"time": [packet.timestamp],
                                time: [packet.timestamp]}
                    )
                ], compat="override", combine_attrs="override")
        return ds


class Ad2cpDataPacket:
    def __init__(self, f: BinaryIO, burst_average_data_record_version: BurstAverageDataRecordVersion):
        self.burst_average_data_record_version = burst_average_data_record_version
        # self.data_record_type: Optional[DataRecordType] = None
        self.data = dict()
        self._read_data_record_header(f)
        self._read_data_record(f)

    @property
    def timestamp(self) -> np.datetime64:
        year = self.data["year"] + 1900
        month = self.data["month"] + 1
        day = self.data["day"]
        hour = self.data["hour"]
        minute = self.data["minute"]
        seconds = self.data["seconds"]
        microsec100 = self.data["microsec100"]
        return np.datetime64(f"{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{seconds:02}.{microsec100:04}")

    def _read_data_record_header(self, f: BinaryIO):
        """
        Reads the header part of the AD2CP packet from the stream
        """

        self.data_record_format = self.HEADER_FORMAT
        raw_header = self._read_data(f, self.data_record_format)
        # don't include the last 2 bytes, which is the header checksum itself
        calculated_checksum = self.checksum(raw_header[: -2])
        expected_checksum = self.data["header_checksum"]
        assert calculated_checksum == expected_checksum, f"invalid header checksum: found {calculated_checksum}, expected {expected_checksum}"

    def _read_data_record(self, f: BinaryIO):
        """
        Reads the data record part of the AD2CP packet from the stream
        """

        if self.data["id"] in (0x15, 0x18):  # burst
            if self.burst_average_data_record_version == BurstAverageDataRecordVersion.VERSION2:
                self.data_record_format = self.BURST_AVERAGE_VERSION2_DATA_RECORD_FORMAT
                self.data_record_type = DataRecordType.BURST_VERSION2
            elif self.burst_average_data_record_version == BurstAverageDataRecordVersion.VERSION3:
                self.data_record_format = self.BURST_AVERAGE_VERSION3_DATA_RECORD_FORMAT
                self.data_record_type = DataRecordType.BURST_VERSION3
            else:
                raise ValueError("invalid burst/average data record version")
        elif self.data["id"] == 0x16:  # average
            if self.burst_average_data_record_version == BurstAverageDataRecordVersion.VERSION2:
                self.data_record_format = self.BURST_AVERAGE_VERSION2_DATA_RECORD_FORMAT
                self.data_record_type = DataRecordType.AVERAGE_VERSION2
            elif self.burst_average_data_record_version == BurstAverageDataRecordVersion.VERSION3:
                self.data_record_format = self.BURST_AVERAGE_VERSION3_DATA_RECORD_FORMAT
                self.data_record_type = DataRecordType.AVERAGE_VERSION3
            else:
                raise ValueError("invalid burst/average data record version")
        elif self.data["id"] in (0x17, 0x1b):  # bottom track
            self.data_record_format = self.BOTTOM_TRACK_DATA_RECORD_FORMAT
            self.data_record_type = DataRecordType.BOTTOM_TRACK
        elif self.data["id"] == 0x1a:  # burst altimeter
            # altimeter is only supported by burst/average version 3
            self.data_record_format = self.BURST_AVERAGE_VERSION3_DATA_RECORD_FORMAT
            self.data_record_type = DataRecordType.BURST_VERSION3
        elif self.data["id"] == 0x1c:  # echosounder
            # echosounder is only supported by burst/average version 3
            self.data_record_format = self.BURST_AVERAGE_VERSION3_DATA_RECORD_FORMAT
            self.data_record_type = DataRecordType.AVERAGE_VERSION3
        elif self.data["id"] == 0x1d:  # dvl water track record
            # TODO: is this correct?
            self.data_record_format = self.BURST_AVERAGE_VERSION3_DATA_RECORD_FORMAT
            self.data_record_type = DataRecordType.AVERAGE_VERSION3
        elif self.data["id"] == 0x1e:  # altimeter
            # altimeter is only supported by burst/average version 3
            self.data_record_format = self.BURST_AVERAGE_VERSION3_DATA_RECORD_FORMAT
            self.data_record_type = DataRecordType.AVERAGE_VERSION3
        elif self.data["id"] == 0x1f:  # average altimeter
            self.data_record_format = self.BURST_AVERAGE_VERSION3_DATA_RECORD_FORMAT
            self.data_record_type = DataRecordType.AVERAGE_VERSION3
        elif self.data["id"] == 0xa0:  # string data
            self.data_record_format = self.STRING_DATA_RECORD_FORMAT
            self.data_record_type = DataRecordType.STRING
        else:
            raise ValueError("invalid data record type id")

        raw_data_record = self._read_data(f, self.data_record_format)
        calculated_checksum = self.checksum(raw_data_record)
        expected_checksum = self.data["data_record_checksum"]
        assert calculated_checksum == expected_checksum, f"invalid data record checksum: found {calculated_checksum}, expected {expected_checksum}"

    def _read_data(self, f: BinaryIO, data_format: List[Field]) -> bytes:
        """
        Reads data from the stream, interpreting the data using the given format
        """

        raw_bytes = bytes()  # combination of all raw fields
        for field_format in data_format:
            field_name = field_format.field_name
            field_entry_size_bytes = field_format.field_entry_size_bytes
            field_entry_data_type = field_format.field_entry_data_type
            field_shape = field_format.field_shape
            field_exists_predicate = field_format.field_exists_predicate
            if not field_exists_predicate(self):
                continue
            if callable(field_entry_size_bytes):
                field_entry_size_bytes = field_entry_size_bytes(self)
            if callable(field_shape):
                field_shape = field_shape(self)

            raw_field = self._read_exact(
                f, field_entry_size_bytes * math.prod(field_shape))
            raw_bytes += raw_field
            if len(field_shape) == 0:
                parsed_field = self._parse(raw_field, field_entry_data_type)
            else:
                # split the field into entries of size field_entry_size_bytes
                raw_field_entries = [raw_field[i * field_entry_size_bytes:(
                    i + 1) * field_entry_size_bytes] for i in range(math.prod(field_shape))]
                # parse each entry individually
                parsed_field_entries = [self._parse(
                    raw_field_entry, field_entry_data_type) for raw_field_entry in raw_field_entries]
                # reshape the list of entries into the correct shape
                parsed_field = np.reshape(parsed_field_entries, field_shape)
            # we cannot check for this before reading because some fields are placeholder fields
            # which, if not read in the correct order with other fields, will offset the rest of the data
            if field_name is not None:
                self.data[field_name] = parsed_field
                self._postprocess(field_name)

        return raw_bytes

    @staticmethod
    def _parse(value: bytes, data_type: DataType) -> Any:
        """
        Parses raw bytes into a value given its data type
        """

        # all numbers are little endian
        if data_type == DataType.RAW_BYTES:
            return value
        elif data_type == DataType.STRING:
            return value.decode("utf-8")
        elif data_type == DataType.SIGNED_INTEGER:
            return int.from_bytes(
                value, byteorder="little", signed=True)
        elif data_type == DataType.UNSIGNED_INTEGER:
            return int.from_bytes(
                value, byteorder="little", signed=False)
        elif data_type == DataType.FLOAT and len(value) == 4:
            return struct.unpack("<f", value)
        elif data_type == DataType.FLOAT and len(value) == 8:
            return struct.unpack("<d", value)
        elif data_type == DataType.SIGNED_FRACTION:
            # TODO: ??????
            pass
        else:
            # unreachable
            raise RuntimeError

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
            last_bytes_read = f.read(
                total_num_bytes_to_read - len(all_bytes_read))
            if len(last_bytes_read) == 0:
                # 0 bytes read with non-0 bytes requested means eof
                raise NoMorePackets
            else:
                all_bytes_read += last_bytes_read
        return all_bytes_read

    def _postprocess(self, field_name):
        """
        Calculates values based on parsed data. This should be called immediately after
        parsing each field in a data record.
        """

        if self.data_record_format == self.BURST_AVERAGE_VERSION2_DATA_RECORD_FORMAT:
            if field_name == "configuration":
                self.data["pressure_sensor_valid"] = self.data["configuration"] & 0b0000_0000_0000_0001 > 0
                self.data["temperature_sensor_valid"] = self.data["configuration"] & 0b0000_0000_0000_0010 > 0
                self.data["compass_sensor_valid"] = self.data["configuration"] & 0b0000_0000_0000_0100 > 0
                self.data["tilt_sensor_valid"] = self.data["configuration"] & 0b0000_0000_0000_1000 > 0
                self.data["velocity_data_included"] = self.data["configuration"] & 0b0000_0000_0010_0000 > 0
                self.data["amplitude_data_included"] = self.data["configuration"] & 0b0000_0000_0100_0000 > 0
                self.data["correlation_data_included"] = self.data["configuration"] & 0b0000_0000_1000_0000 > 0
            elif field_name == "num_beams_and_coordinate_system_and_num_cells":
                self.data["num_cells"] = self.data["num_beams_and_coordinate_system_and_num_cells"] & 0b0000_0011_1111_1111
                self.data["coordinate_system"] = (
                    self.data["num_beams_and_coordinate_system_and_num_cells"] & 0b0000_1100_0000_0000) >> 10
                self.data["num_beams"] = (
                    self.data["num_beams_and_coordinate_system_and_num_cells"] & 0b1111_0000_0000_0000) >> 12
        elif self.data_record_format == self.BURST_AVERAGE_VERSION3_DATA_RECORD_FORMAT:
            if field_name == "configuration":
                self.data["pressure_sensor_valid"] = self.data["configuration"] & 0b0000_0000_0000_0001 > 0
                self.data["temperature_sensor_valid"] = self.data["configuration"] & 0b0000_0000_0000_0010 > 0
                self.data["compass_sensor_valid"] = self.data["configuration"] & 0b0000_0000_0000_0100 > 0
                self.data["tilt_sensor_valid"] = self.data["configuration"] & 0b0000_0000_0000_1000 > 0
                self.data["velocity_data_included"] = self.data["configuration"] & 0b0000_0000_0010_0000 > 0
                self.data["amplitude_data_included"] = self.data["configuration"] & 0b0000_0000_0100_0000 > 0
                self.data["correlation_data_included"] = self.data["configuration"] & 0b0000_0000_1000_0000 > 0
                self.data["altimeter_data_included"] = self.data["configuration"] & 0b0000_0001_0000_0000 > 0
                self.data["altimeter_raw_data_included"] = self.data["configuration"] & 0b0000_0010_0000_0000 > 0
                self.data["ast_data_included"] = self.data["configuration"] & 0b0000_0100_0000_0000 > 0
                self.data["echo_sounder_data_included"] = self.data["configuration"] & 0b0000_1000_0000_0000 > 0
                self.data["ahrs_data_included"] = self.data["configuration"] & 0b0001_0000_0000_0000 > 0
                self.data["percentage_good_data_included"] = self.data["configuration"] & 0b0010_0000_0000_0000 > 0
                self.data["std_dev_data_included"] = self.data["configuration"] & 0b0100_0000_0000_0000 > 0
            elif field_name == "num_beams_and_coordinate_system_and_num_cells":
                if self.data["echo_sounder_data_included"]:
                    self.data["num_echo_sounder_cells"] = self.data["num_beams_and_coordinate_system_and_num_cells"]
                else:
                    self.data["num_cells"] = self.data["num_beams_and_coordinate_system_and_num_cells"] & 0b0000_0011_1111_1111
                    self.data["coordinate_system"] = (
                        self.data["num_beams_and_coordinate_system_and_num_cells"] & 0b0000_1100_0000_0000) >> 10
                    self.data["num_beams"] = (
                        self.data["num_beams_and_coordinate_system_and_num_cells"] & 0b1111_0000_0000_0000) >> 12
            elif field_name == "ambiguity_velocity_or_echo_sounder_frequency":
                if self.data["echo_sounder_data_included"]:
                    # This is specified as "echo sounder frequency", but the description technically
                    # says "number of echo sounder cells". It is probably the frequency and not the number of cells
                    # because the number of cells already replaces the data in "num_beams_and_coordinate_system_and_num_cells"
                    # when an echo sounder is present
                    self.data["echo_sounder_frequency"] = self.data["ambiguity_velocity_or_echo_sounder_frequency"]
                else:
                    self.data["ambiguity_velocity"] = self.data["ambiguity_velocity_or_echo_sounder_frequency"]
        elif self.data_record_format == self.BOTTOM_TRACK_DATA_RECORD_FORMAT:
            if field_name == "configuration":
                self.data["pressure_sensor_valid"] = self.data["data"]["configuration"] & 0b0000_0000_0000_0001 > 0
                self.data["temperature_sensor_valid"] = self.data["data"]["configuration"] & 0b0000_0000_0000_0010 > 0
                self.data["compass_sensor_valid"] = self.data["data"]["configuration"] & 0b0000_0000_0000_0100 > 0
                self.data["tilt_sensor_valid"] = self.data["data"]["configuration"] & 0b0000_0000_0000_1000 > 0
                self.data["velocity_data_included"] = self.data["data"]["configuration"] & 0b0000_0000_0010_0000 > 0
                self.data["distance_data_included"] = self.data["data"]["configuration"] & 0b0000_0001_0000_0000 > 0
                self.data["figure_of_merit_data_included"] = self.data["data"]["configuration"] & 0b0000_0010_0000_0000 > 0
            elif field_name == "num_beams_and_coordinate_system_and_num_cells":
                self.data["num_cells"] = self.data["num_beams_and_coordinate_system_and_num_cells"] & 0b0000_0011_1111_1111
                self.data["coordinate_system"] = (
                    self.data["num_beams_and_coordinate_system_and_num_cells"] & 0b0000_1100_0000_0000) >> 10
                self.data["num_beams"] = (
                    self.data["num_beams_and_coordinate_system_and_num_cells"] & 0b1111_0000_0000_0000) >> 12

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

    HEADER_FORMAT: List[Field] = [
        F("sync", 1, UNSIGNED_INTEGER),
        F("header_size", 1, UNSIGNED_INTEGER),
        F("id", 1, UNSIGNED_INTEGER),
        F("family", 1, UNSIGNED_INTEGER),
        F("data_record_size", 2, UNSIGNED_INTEGER),
        F("data_record_checksum", 2, UNSIGNED_INTEGER),
        F("header_checksum", 2, UNSIGNED_INTEGER),
    ]
    STRING_DATA_RECORD_FORMAT: List[Field] = [
        F("string_data_id", 1, UNSIGNED_INTEGER),
        F("string_data",
          lambda self: self.data["data_record_size"] - 1, STRING),
    ]
    BURST_AVERAGE_VERSION2_DATA_RECORD_FORMAT: List[Field] = [
        F("version", 1, UNSIGNED_INTEGER),
        F("offset_of_data", 1, UNSIGNED_INTEGER),
        F("serial_number", 4, UNSIGNED_INTEGER),
        F("configuration", 2, UNSIGNED_INTEGER),
        F("year", 1, UNSIGNED_INTEGER),
        F("month", 1, UNSIGNED_INTEGER),
        F("day", 1, UNSIGNED_INTEGER),
        F("hour", 1, UNSIGNED_INTEGER),
        F("minute", 1, UNSIGNED_INTEGER),
        F("seconds", 1, UNSIGNED_INTEGER),
        F("microsec100", 2, UNSIGNED_INTEGER),
        F("speed_of_sound", 2, UNSIGNED_INTEGER),
        F("temperature", 2, SIGNED_INTEGER),
        F("pressure", 4, UNSIGNED_INTEGER),
        F("heading", 2, UNSIGNED_INTEGER),
        F("pitch", 2, SIGNED_INTEGER),
        F("roll", 2, SIGNED_INTEGER),
        F("error", 2, UNSIGNED_INTEGER),
        F("status", 2, UNSIGNED_INTEGER),
        F("num_beams_and_coordinate_system_and_num_cells", 2, UNSIGNED_INTEGER),
        F("cell_size", 2, UNSIGNED_INTEGER),
        F("blanking", 2, UNSIGNED_INTEGER),
        F("velocity_range", 2, UNSIGNED_INTEGER),
        F("battery_voltage", 2, UNSIGNED_INTEGER),
        F("magnetometer_raw_x", 2, SIGNED_INTEGER),
        F("magnetometer_raw_y", 2, SIGNED_INTEGER),
        F("magnetometer_raw_z", 2, SIGNED_INTEGER),
        F("accelerometer_raw_x_axis", 2, SIGNED_INTEGER),
        F("accelerometer_raw_y_axis", 2, SIGNED_INTEGER),
        F("accelerometer_raw_z_axis", 2, SIGNED_INTEGER),
        F("ambiguity_velocity", 2, UNSIGNED_INTEGER),
        F("dataset_description", 2, UNSIGNED_INTEGER),
        F("transmit_energy", 2, UNSIGNED_INTEGER),
        F("velocity_scaling", 1, SIGNED_INTEGER),
        F("power_level", 1, SIGNED_INTEGER),
        F(None, 4, UNSIGNED_INTEGER),
        F(
            "velocity_data",
            2,
            SIGNED_INTEGER,
            field_shape=lambda self: [
                self.data["num_beams"], self.data["num_cells"]],
            field_exists_predicate=lambda self: self.data["velocity_data_included"]
        ),
        F(
            "amplitude_data",
            1,
            UNSIGNED_INTEGER,
            field_shape=lambda self: [
                self.data["num_beams"], self.data["num_cells"]],
            field_exists_predicate=lambda self: self.data["amplitude_data_included"]
        ),
        F(
            "correlation_data",
            1,
            UNSIGNED_INTEGER,
            field_shape=lambda self: [
                self.data["num_beams"], self.data["num_cells"]],
            field_exists_predicate=lambda self: self.data["correlation_data_included"]
        )
    ]
    BURST_AVERAGE_VERSION3_DATA_RECORD_FORMAT: List[Field] = [
        F("version", 1, UNSIGNED_INTEGER),
        F("offset_of_data", 1, UNSIGNED_INTEGER),
        F("configuration", 2, UNSIGNED_INTEGER),
        F("serial_number", 4, UNSIGNED_INTEGER),
        F("year", 1, UNSIGNED_INTEGER),
        F("month", 1, UNSIGNED_INTEGER),
        F("day", 1, UNSIGNED_INTEGER),
        F("hour", 1, UNSIGNED_INTEGER),
        F("minute", 1, UNSIGNED_INTEGER),
        F("seconds", 1, UNSIGNED_INTEGER),
        F("microsec100", 2, UNSIGNED_INTEGER),
        F("speed_of_sound", 2, UNSIGNED_INTEGER),
        F("temperature", 2, SIGNED_INTEGER),
        F("pressure", 4, UNSIGNED_INTEGER),
        F("heading", 2, UNSIGNED_INTEGER),
        F("pitch", 2, SIGNED_INTEGER),
        F("roll", 2, SIGNED_INTEGER),
        F("num_beams_and_coordinate_system_and_num_cells", 2, UNSIGNED_INTEGER),
        F("cell_size", 2, UNSIGNED_INTEGER),
        F("blanking", 2, UNSIGNED_INTEGER),
        F("nominal_correlation", 1, UNSIGNED_INTEGER),
        F("temperature_from_pressure_sensor", 1, UNSIGNED_INTEGER),
        F("battery_voltage", 2, UNSIGNED_INTEGER),
        F("magnetometer_raw_x", 2, SIGNED_INTEGER),
        F("magnetometer_raw_y", 2, SIGNED_INTEGER),
        F("magnetometer_raw_z", 2, SIGNED_INTEGER),
        F("accelerometer_raw_x_axis", 2, SIGNED_INTEGER),
        F("accelerometer_raw_y_axis", 2, SIGNED_INTEGER),
        F("accelerometer_raw_z_axis", 2, SIGNED_INTEGER),
        F("ambiguity_velocity_or_echo_sounder_frequency", 2, UNSIGNED_INTEGER),
        F("dataset_description", 2, UNSIGNED_INTEGER),
        F("transmit_energy", 2, UNSIGNED_INTEGER),
        F("velocity_scaling", 1, SIGNED_INTEGER),
        F("power_level", 1, SIGNED_INTEGER),
        F("magnetometer_temperature", 2, SIGNED_INTEGER),
        F("real_time_clock_temperature", 2, SIGNED_INTEGER),
        F("error", 2, UNSIGNED_INTEGER),
        F("status0", 2, UNSIGNED_INTEGER),
        F("status", 4, UNSIGNED_INTEGER),
        F("ensemble_counter", 4, UNSIGNED_INTEGER),
        F(
            "velocity_data",
            2,
            SIGNED_INTEGER,
            field_shape=lambda self: [
                self.data["num_beams"], self.data["num_cells"]],
            field_exists_predicate=lambda self: self.data["velocity_data_included"]
        ),
        F(
            "amplitude_data",
            1,
            UNSIGNED_INTEGER,
            field_shape=lambda self: [
                self.data["num_beams"], self.data["num_cells"]],
            field_exists_predicate=lambda self: self.data["amplitude_data_included"]
        ),
        F(
            "correlation_data",
            1,
            UNSIGNED_INTEGER,
            field_shape=lambda self: [
                self.data["num_beams"], self.data["num_cells"]],
            field_exists_predicate=lambda self: self.data["correlation_data_included"]
        ),
        F("altimeter_distance", 4, FLOAT,
          field_exists_predicate=lambda self: self.data["altimeter_data_included"]),
        F("altimeter_quality", 2, UNSIGNED_INTEGER,
          field_exists_predicate=lambda self: self.data["altimeter_data_included"]),
        F("ast_distance", 4, FLOAT,
          field_exists_predicate=lambda self: self.data["ast_data_included"]),
        F("ast_quality", 2, UNSIGNED_INTEGER,
          field_exists_predicate=lambda self: self.data["ast_data_included"]),
        F("ast_offset_10us", 2, SIGNED_INTEGER,
          field_exists_predicate=lambda self: self.data["ast_data_included"]),
        F("ast_pressure", 4, FLOAT,
          field_exists_predicate=lambda self: self.data["ast_data_included"]),
        F("altimeter_spare", 1, RAW_BYTES, field_shape=[8], field_exists_predicate=lambda self: self.data["ast_data_included"]
          ),
        F(
            "altimeter_raw_data_num_samples",
            # The field size of this field is technically specified as number of samples * 2,
            # but seeing as the field is called "num samples," and the field which is supposed
            # to contain the samples is specified as having a constant size of 2, these fields
            # sizes were likely incorrectly swapped.
            2,
            UNSIGNED_INTEGER,
            field_exists_predicate=lambda self: self.data["altimeter_raw_data_included"]
        ),
        F("altimeter_raw_data_sample_distance", 2, UNSIGNED_INTEGER,
          field_exists_predicate=lambda self: self.data["altimeter_raw_data_included"]),
        F(
            "altimeter_raw_data_samples",
            2,
            SIGNED_FRACTION,
            field_shape=lambda self: [
                self.data["altimeter_raw_data_num_samples"]],
            field_exists_predicate=lambda self: self.data["altimeter_raw_data_included"],
        ),
        F(
            "echo_sounder_data",
            2,
            UNSIGNED_INTEGER,
            field_shape=lambda self: [self.data["num_cells"]],
            field_exists_predicate=lambda self: self.data["echo_sounder_data_included"]
        ),
        F("ahrs_rotation_matrix_m11", 4, FLOAT,
          field_exists_predicate=lambda self: self.data["ahrs_data_included"]),
        F("ahrs_rotation_matrix_m12", 4, FLOAT,
          field_exists_predicate=lambda self: self.data["ahrs_data_included"]),
        F("ahrs_rotation_matrix_m13", 4, FLOAT,
          field_exists_predicate=lambda self: self.data["ahrs_data_included"]),
        F("ahrs_rotation_matrix_m21", 4, FLOAT,
          field_exists_predicate=lambda self: self.data["ahrs_data_included"]),
        F("ahrs_rotation_matrix_m22", 4, FLOAT,
          field_exists_predicate=lambda self: self.data["ahrs_data_included"]),
        F("ahrs_rotation_matrix_m23", 4, FLOAT,
          field_exists_predicate=lambda self: self.data["ahrs_data_included"]),
        F("ahrs_rotation_matrix_m31", 4, FLOAT,
          field_exists_predicate=lambda self: self.data["ahrs_data_included"]),
        F("ahrs_rotation_matrix_m32", 4, FLOAT,
          field_exists_predicate=lambda self: self.data["ahrs_data_included"]),
        F("ahrs_rotation_matrix_m33", 4, FLOAT,
          field_exists_predicate=lambda self: self.data["ahrs_data_included"]),
        F("ahrs_quaternions_w", 4, FLOAT,
          field_exists_predicate=lambda self: self.data["ahrs_data_included"]),
        F("ahrs_quaternions_x", 4, FLOAT,
          field_exists_predicate=lambda self: self.data["ahrs_data_included"]),
        F("ahrs_quaternions_y", 4, FLOAT,
          field_exists_predicate=lambda self: self.data["ahrs_data_included"]),
        F("ahrs_quaternions_z", 4, FLOAT,
          field_exists_predicate=lambda self: self.data["ahrs_data_included"]),
        F("ahrs_gyro_x", 4, FLOAT,
          field_exists_predicate=lambda self: self.data["ahrs_data_included"]),
        F("ahrs_gyro_y", 4, FLOAT,
          field_exists_predicate=lambda self: self.data["ahrs_data_included"]),
        F("ahrs_gyro_z", 4, FLOAT,
          field_exists_predicate=lambda self: self.data["ahrs_data_included"]),
        # ("ahrs_gyro", 4, FLOAT, [3], lambda self: self.data["ahrs_data_included"]),
        F(
            "percentage_good_data",
            1,
            UNSIGNED_INTEGER,
            field_shape=lambda self: [self.data["num_cells"]],
            field_exists_predicate=lambda self: self.data["percentage_good_data_included"]
        ),
        # only the pitch field is labeled as included when the "std dev data included"
        # bit is set, but this is likely a mistake
        F("std_dev_pitch", 2, SIGNED_INTEGER,
          field_exists_predicate=lambda self: self.data["std_dev_data_included"]),
        F("std_dev_roll", 2, SIGNED_INTEGER,
          field_exists_predicate=lambda self: self.data["std_dev_data_included"]),
        F("std_dev_heading", 2, SIGNED_INTEGER,
          field_exists_predicate=lambda self: self.data["std_dev_data_included"]),
        F("std_dev_pressure", 2, SIGNED_INTEGER,
          field_exists_predicate=lambda self: self.data["std_dev_data_included"]),
        F(None, 24, RAW_BYTES,
          field_exists_predicate=lambda self: self.data["std_dev_data_included"])
    ]
    BOTTOM_TRACK_DATA_RECORD_FORMAT: List[Field] = [
        F("version", 1, UNSIGNED_INTEGER),
        F("offset_of_data", 1, UNSIGNED_INTEGER),
        F("configuration", 2, UNSIGNED_INTEGER),
        F("serial_number", 4, UNSIGNED_INTEGER),
        F("year", 1, UNSIGNED_INTEGER),
        F("month", 1, UNSIGNED_INTEGER),
        F("day", 1, UNSIGNED_INTEGER),
        F("hour", 1, UNSIGNED_INTEGER),
        F("minute", 1, UNSIGNED_INTEGER),
        F("seconds", 1, UNSIGNED_INTEGER),
        F("microsec100", 2, UNSIGNED_INTEGER),
        F("speed_of_sound", 2, UNSIGNED_INTEGER),
        F("temperature", 2, SIGNED_INTEGER),
        F("pressure", 4, UNSIGNED_INTEGER),
        F("heading", 2, UNSIGNED_INTEGER),
        F("pitch", 2, SIGNED_INTEGER),
        F("roll", 2, SIGNED_INTEGER),
        F("num_beams_and_coordinate_system_and_num_cells", 2, UNSIGNED_INTEGER),
        F("cell_size", 2, UNSIGNED_INTEGER),
        F("blanking", 2, UNSIGNED_INTEGER),
        F("nominal_correlation", 1, UNSIGNED_INTEGER),
        F(None, 1, RAW_BYTES),
        F("battery_voltage", 2, UNSIGNED_INTEGER),
        F("magnetometer_raw_x", 2, SIGNED_INTEGER),
        F("magnetometer_raw_y", 2, SIGNED_INTEGER),
        F("magnetometer_raw_z", 2, SIGNED_INTEGER),
        F("accelerometer_raw_x_axis", 2, SIGNED_INTEGER),
        F("accelerometer_raw_y_axis", 2, SIGNED_INTEGER),
        F("accelerometer_raw_z_axis", 2, SIGNED_INTEGER),
        F("ambiguity_velocity", 4, UNSIGNED_INTEGER),
        F("dataset_description", 2, UNSIGNED_INTEGER),
        F("transmit_energy", 2, UNSIGNED_INTEGER),
        F("velocity_scaling", 1, SIGNED_INTEGER),
        F("power_level", 1, SIGNED_INTEGER),
        F("magnetometer_temperature", 2, SIGNED_INTEGER),
        F("real_time_clock_temperature", 2, SIGNED_INTEGER),
        F("error", 4, UNSIGNED_INTEGER),
        F("status", 4, UNSIGNED_INTEGER),
        F("ensemble_counter", 4, UNSIGNED_INTEGER),
        F(
            "velocity_data",
            4,
            SIGNED_INTEGER,
            field_shape=lambda self: [self.data["num_beams"]],
            field_exists_predicate=lambda self: self.data["velocity_data_included"]
        ),
        F(
            "distance_data",
            4,
            SIGNED_INTEGER,
            field_shape=lambda self: [self.data["num_beams"]],
            field_exists_predicate=lambda self: self.data["distance_data_included"]
        ),
        F(
            "figure_of_merit_data",
            2,
            UNSIGNED_INTEGER,
            field_shape=lambda self: [self.data["num_beams"]],
            field_exists_predicate=lambda self: self.data["figure_of_merit_data_included"]
        )
    ]


if __name__ == "__main__":
    import sys
    with open(sys.argv[1], "rb") as f:
        reader = Ad2cpReader(f)

    x = SetGroupsAd2cp(reader, "test.nc")
    x.save()
