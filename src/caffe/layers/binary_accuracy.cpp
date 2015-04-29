#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void BinaryAccuracyLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void BinaryAccuracyLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->num(), bottom[1]->num())
      << "The data and label should have the same number.";
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels())
      << "The data and label should have the same channels.";
  CHECK_EQ(bottom[0]->height(), bottom[1]->height())
      << "The data and label should have the same height.";
  CHECK_EQ(bottom[0]->width(), bottom[1]->width())
      << "The data and label should have the same width.";
  top[0]->Reshape(1, 1, 1, 1);
}

template <typename Dtype>
void BinaryAccuracyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Dtype accuracy = 0;
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  int num = bottom[0]->count();

  for (int i = 0; i < num; ++i) {
    int prediction = static_cast<int>(bottom_data[i] > 0);
    int label = static_cast<int>(bottom_label[i]);
    int match = static_cast<int>(prediction == label);
    accuracy += match;
  }

  // LOG(INFO) << "Accuracy: " << accuracy;
  top[0]->mutable_cpu_data()[0] = accuracy / num;
  // Accuracy layer should not be used as a loss function.
}

INSTANTIATE_CLASS(BinaryAccuracyLayer);
REGISTER_LAYER_CLASS(BIN_ACCURACY, BinaryAccuracyLayer);
}  // namespace caffe
