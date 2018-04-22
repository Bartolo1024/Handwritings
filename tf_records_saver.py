import tensorflow as tf
import os
import cv2
import forms_txt_parse as txt
import numpy as np
from xml_parse import coordinates_generator


class RecordsSaver(object):

    default_filename = os.path.join(os.path.split(os.path.abspath(__file__))[0], "tf_records/img_box.tfrecords")

    def __init__(self, filename=default_filename):
        self.writer = tf.python_io.TFRecordWriter(filename)

    @staticmethod
    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    @staticmethod
    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def write_image(self, image, label):
        rows = image.shape[0]
        cols = image.shape[1]
        image_raw = image.tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'image_raw': self._bytes_feature(image_raw),
            'height': self._int64_feature(rows),
            'width': self._int64_feature(cols),
            'label': self._int64_feature(int(label))
        }))
        self.writer.write(example.SerializeToString())

    def close(self):
        self.writer.close()


if __name__ == '__main__':
    saver = RecordsSaver()
    df = txt.read_forms_dataframe()

    for idx, row in df.iterrows():
        print(row['image_path'])
        print(row['xml_path'])
        img = cv2.imread(row['image_path'])
        for box in coordinates_generator(row['xml_path'], [500, 150], np.shape(img)):
            saver.write_image(img[box[0]: box[2], box[1]: box[3]], row['writer_id'])

    saver.close()
