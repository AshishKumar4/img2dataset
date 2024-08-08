""""writer module handle writing the images to disk"""

import json
import os

import fsspec
import numpy as np
import pyarrow as pa
import pyarrow.fs
import pyarrow.parquet as pq
import webdataset as wds
import struct
from array_record.python.array_record_module import ArrayRecordWriter

class BufferedParquetWriter:
    """Write samples to parquet files incrementally with a buffer"""

    def __init__(self, output_file, schema, buffer_size=100):
        self.buffer_size = buffer_size
        self.schema = schema
        self._initiatlize_buffer()
        fs, output_path = fsspec.core.url_to_fs(output_file)
        self.output_fd = fs.open(output_path, "wb")
        self.parquet_writer = pq.ParquetWriter(self.output_fd, schema)

    def _initiatlize_buffer(self):
        self.current_buffer_size = 0
        self.buffer = {k: [] for k in self.schema.names}

    def _add_sample_to_buffer(self, sample):
        for k in self.schema.names:
            self.buffer[k].append(sample[k])
        self.current_buffer_size += 1

    def write(self, sample):
        if self.current_buffer_size >= self.buffer_size:
            self.flush()
        self._add_sample_to_buffer(sample)

    def flush(self):
        """Write the buffer to disk"""
        if self.current_buffer_size == 0:
            return

        df = pa.Table.from_pydict(self.buffer, self.schema)
        self.parquet_writer.write_table(df)
        self._initiatlize_buffer()

    def close(self):
        self.flush()
        if self.parquet_writer is not None:
            self.parquet_writer.close()
            self.parquet_writer = None
            self.output_fd.close()


class ParquetSampleWriter:
    """ParquetSampleWriter is a image+caption writer to parquet"""

    def __init__(
        self,
        shard_id,
        output_folder,
        save_caption,
        oom_shard_count,
        schema,
        encode_format,
    ):
        self.oom_shard_count = oom_shard_count
        self.encode_format = encode_format
        schema = schema.append(pa.field(encode_format, pa.binary()))
        shard_name = "{shard_id:0{oom_shard_count}d}".format(  # pylint: disable=consider-using-f-string
            shard_id=shard_id, oom_shard_count=oom_shard_count
        )
        output_file = f"{output_folder}/{shard_name}.parquet"
        self.buffered_parquet_writer = BufferedParquetWriter(output_file, schema, 100)
        self.save_caption = save_caption

    def write(self, img_str, key, caption, meta):
        """Keep sample in memory then write to disk when close() is called"""
        if img_str is not None:
            sample = {"key": key, self.encode_format: img_str}
            if self.save_caption:
                sample["txt"] = str(caption) if caption is not None else ""
        else:
            sample = {"key": key, self.encode_format: None}
            if self.save_caption:
                sample["txt"] = None
        sample.update(meta)
        self.buffered_parquet_writer.write(sample)

    def close(self):
        self.buffered_parquet_writer.close()


class WebDatasetSampleWriter:
    """WebDatasetSampleWriter is a image+caption writer to webdataset"""

    def __init__(
        self,
        shard_id,
        output_folder,
        save_caption,
        oom_shard_count,
        schema,
        encode_format,
    ):
        self.oom_shard_count = oom_shard_count
        shard_name = "{shard_id:0{oom_shard_count}d}".format(  # pylint: disable=consider-using-f-string
            shard_id=shard_id, oom_shard_count=oom_shard_count
        )
        self.shard_id = shard_id
        fs, output_path = fsspec.core.url_to_fs(output_folder)
        self.tar_fd = fs.open(f"{output_path}/{shard_name}.tar", "wb")
        self.tarwriter = wds.TarWriter(self.tar_fd)
        self.save_caption = save_caption
        self.buffered_parquet_writer = BufferedParquetWriter(output_folder + "/" + shard_name + ".parquet", schema, 100)
        self.encode_format = encode_format

    def write(self, img_str, key, caption, meta):
        """write sample to tars"""
        if img_str is not None:
            sample = {"__key__": key, self.encode_format: img_str}
            if self.save_caption:
                sample["txt"] = str(caption) if caption is not None else ""
            # some meta data may not be JSON serializable
            for k, v in meta.items():
                if isinstance(v, np.ndarray):
                    meta[k] = v.tolist()
            sample["json"] = json.dumps(meta, indent=4)
            self.tarwriter.write(sample)
        self.buffered_parquet_writer.write(meta)

    def close(self):
        self.buffered_parquet_writer.close()
        self.tarwriter.close()
        self.tar_fd.close()

class TFRecordSampleWriter:
    """TFRecordSampleWriter is a image+caption writer to TFRecord"""

    def __init__(
        self,
        shard_id,
        output_folder,
        save_caption,
        oom_shard_count,
        schema,
        encode_format,
    ):
        try:
            os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
            import tensorflow_io as _  # pylint: disable=import-outside-toplevel
            from tensorflow.python.lib.io.tf_record import TFRecordWriter  # pylint: disable=import-outside-toplevel
            from tensorflow.python.training.training import (  # pylint: disable=import-outside-toplevel
                BytesList,
                Example,
                Feature,
                Features,
                FloatList,
                Int64List,
            )

            self._BytesList = BytesList  # pylint: disable=invalid-name
            self._Int64List = Int64List  # pylint: disable=invalid-name
            self._FloatList = FloatList  # pylint: disable=invalid-name
            self._Example = Example  # pylint: disable=invalid-name
            self._Features = Features  # pylint: disable=invalid-name
            self._Feature = Feature  # pylint: disable=invalid-name
        except ImportError as e:
            raise ModuleNotFoundError(
                "tfrecords require tensorflow and tensorflow_io to be installed."
                "Run `pip install tensorflow tensorflow_io`."
            ) from e

        self.oom_shard_count = oom_shard_count
        shard_name = "{shard_id:0{oom_shard_count}d}".format(  # pylint: disable=consider-using-f-string
            shard_id=shard_id, oom_shard_count=oom_shard_count
        )
        self.shard_id = shard_id
        self.tf_writer = TFRecordWriter(f"{output_folder}/{shard_name}.tfrecord")
        self.save_caption = save_caption
        self.buffered_parquet_writer = BufferedParquetWriter(output_folder + "/" + shard_name + ".parquet", schema, 100)
        self.encode_format = encode_format

    def write(self, img_str, key, caption, meta):
        """Write a sample using tfrecord writer"""
        if img_str is not None:
            sample = {
                "key": self._bytes_feature(key.encode()),
                self.encode_format: self._bytes_feature(img_str),
            }
            if self.save_caption:
                sample["txt"] = self._bytes_feature(str(caption) if caption is not None else "")
            for k, v in meta.items():
                sample[k] = self._feature(v)
            tf_example = self._Example(features=self._Features(feature=sample))
            self.tf_writer.write(tf_example.SerializeToString())
        self.buffered_parquet_writer.write(meta)

    def close(self):
        self.buffered_parquet_writer.close()
        self.tf_writer.close()

    def _feature(self, value):
        """Convert to proper feature type"""
        if isinstance(value, list):
            return self._list_feature(value)
        elif isinstance(value, int):
            return self._int64_feature(value)
        elif isinstance(value, float):
            return self._float_feature(value)
        else:
            return self._bytes_feature(value)

    def _bytes_feature(self, value):
        """Returns a bytes_list from a string / byte."""
        if value is None:
            value = ""
        if isinstance(value, str):
            value = value.encode()
        return self._Feature(bytes_list=self._BytesList(value=[value]))

    def _float_feature(self, value):
        """Returns a float_list from a float / double."""
        return self._Feature(float_list=self._FloatList(value=[value]))

    def _int64_feature(self, value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return self._Feature(int64_list=self._Int64List(value=[value]))

    def _list_feature(self, value):
        """Returns an list of int64_list, float_list, bytes_list."""
        if isinstance(value[0], int):
            return self._Feature(int64_list=self._Int64List(value=value))
        elif isinstance(value[0], float):
            return self._Feature(float_list=self._FloatList(value=value))
        else:
            for i, bytes_feature in enumerate(value):
                if bytes_feature is None:
                    value[i] = ""
                if isinstance(bytes_feature, str):
                    value[i] = bytes_feature.encode()
            return self._Feature(bytes_list=self._BytesList(value=value))


class FilesSampleWriter:
    """FilesSampleWriter is a caption+image writer to files"""

    def __init__(
        self,
        shard_id,
        output_folder,
        save_caption,
        oom_shard_count,
        schema,
        encode_format,
    ):
        self.oom_shard_count = oom_shard_count
        shard_name = "{shard_id:0{oom_shard_count}d}".format(  # pylint: disable=consider-using-f-string
            shard_id=shard_id, oom_shard_count=oom_shard_count
        )
        self.shard_id = shard_id
        self.fs, self.subfolder = fsspec.core.url_to_fs(f"{output_folder}/{shard_name}")
        if not self.fs.exists(self.subfolder):
            self.fs.mkdir(self.subfolder)
        self.save_caption = save_caption
        self.buffered_parquet_writer = BufferedParquetWriter(output_folder + "/" + shard_name + ".parquet", schema, 100)
        self.encode_format = encode_format

    def write(self, img_str, key, caption, meta):
        """Write sample to disk"""
        if img_str is not None:
            filename = f"{self.subfolder}/{key}.{self.encode_format}"
            with self.fs.open(filename, "wb") as f:
                f.write(img_str)
            if self.save_caption:
                caption = str(caption) if caption is not None else ""
                caption_filename = f"{self.subfolder}/{key}.txt"
                with self.fs.open(caption_filename, "w") as f:
                    f.write(str(caption))

            # some meta data may not be JSON serializable
            for k, v in meta.items():
                if isinstance(v, np.ndarray):
                    meta[k] = v.tolist()
            j = json.dumps(meta, indent=4)
            meta_filename = f"{self.subfolder}/{key}.json"
            with self.fs.open(meta_filename, "w") as f:
                f.write(j)
        self.buffered_parquet_writer.write(meta)

    def close(self):
        self.buffered_parquet_writer.close()
import struct 

def pack_dict_of_byte_arrays(data_dict):
    """
    Pack a dictionary of byte arrays into a single byte array.
    
    Args:
        data_dict (dict): Dictionary where keys are strings and values are byte arrays.
        
    Returns:
        bytes: Packed byte array.
    """
    packed_data = bytearray()
    
    for key, byte_array in data_dict.items():
        # Ensure the key is a string
        if not isinstance(key, str):
            raise ValueError("Keys must be strings")
        
        # Convert the key to bytes
        key_bytes = key.encode('utf-8')
        
        # Pack the key length and key bytes
        packed_data.extend(struct.pack('I', len(key_bytes)))
        packed_data.extend(key_bytes)
        
        # Pack the byte array length and byte array
        packed_data.extend(struct.pack('I', len(byte_array)))
        packed_data.extend(byte_array)
    
    return bytes(packed_data)

def unpack_dict_of_byte_arrays(packed_data):
    """
    Unpack a single byte array into a dictionary of byte arrays.
    
    Args:
        packed_data (bytes): Packed byte array.
        
    Returns:
        dict: Dictionary where keys are strings and values are byte arrays.
    """
    unpacked_dict = {}
    offset = 0
    
    while offset < len(packed_data):
        # Unpack the key length
        key_length = struct.unpack_from('I', packed_data, offset)[0]
        offset += struct.calcsize('I')
        
        # Unpack the key bytes and convert to string
        key = packed_data[offset:offset+key_length].decode('utf-8')
        offset += key_length
        
        # Unpack the byte array length
        byte_array_length = struct.unpack_from('I', packed_data, offset)[0]
        offset += struct.calcsize('I')
        
        # Unpack the byte array
        byte_array = packed_data[offset:offset+byte_array_length]
        offset += byte_array_length
        
        unpacked_dict[key] = byte_array
    
    return unpacked_dict

class ArrayRecordSampleWriter:
    """ArrayRecordSampleWriter is a writer to ArrayRecord format"""

    def __init__(
        self,
        shard_id,
        output_folder,
        save_caption,
        oom_shard_count,
        schema,
        encode_format,
    ):
        self.oom_shard_count = oom_shard_count
        self.encode_format = encode_format
        self.save_caption = save_caption
        shard_name = "{shard_id:0{oom_shard_count}d}".format(  # pylint: disable=consider-using-f-string
            shard_id=shard_id, oom_shard_count=oom_shard_count
        )
        # self.buffered_parquet_writer = BufferedParquetWriter(output_folder + "/" + shard_name + ".parquet", schema, 100)
        self.output_file = f"{output_folder}/{shard_name}.array_record"
        if "gs:" in output_folder:
            self.tmp_file = f'/tmp/{shard_name}.array_record'
        else:
            self.tmp_file = self.output_file
        self.writer = ArrayRecordWriter(self.tmp_file, options=f"group_size:1")
        
    def write(self, img_str, key, caption, meta):
        """Write sample to ArrayRecord"""
        if img_str is not None:
            sample = {
                "key": self._bytes_feature(key.encode()),
                self.encode_format: self._bytes_feature(img_str),
            }
            if self.save_caption:
                sample["txt"] = caption.encode() if caption is not None else b""
            for k, v in meta.items():
                if isinstance(v, np.ndarray):
                    meta[k] = v.tolist()
            sample["meta"] = json.dumps(meta).encode()
            self.writer.write(pack_dict_of_byte_arrays(sample))
        # self.buffered_parquet_writer.write(meta)
            
    def _bytes_feature(self, value):
        if value is None:
            value = ""
        if isinstance(value, str):
            value = value.encode()
        return value

    def close(self):
        self.writer.close()
        # self.buffered_parquet_writer.close()
        if self.tmp_file != self.output_file:
            pyarrow.fs.copy_files(self.tmp_file, self.output_file, chunk_size=2**24)
        os.remove(self.tmp_file)

class DummySampleWriter:
    """Does not write"""

    def __init__(self, shard_id, output_folder, save_caption, oom_shard_count, schema, encode_format):
        pass

    def write(self, img_str, key, caption, meta):
        pass

    def close(self):
        pass
