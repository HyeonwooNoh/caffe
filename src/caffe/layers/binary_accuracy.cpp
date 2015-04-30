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
  Dtype num_tp = 0; // true positives
  Dtype num_fn = 0; // false negatives
  Dtype num_tn = 0; // true negatives
  Dtype num_fp = 0; // false positives

  for (int i = 0; i < num; ++i) {
    int prediction = static_cast<int>(bottom_data[i] > 0);
    int label = static_cast<int>(bottom_label[i]);
    int tp = static_cast<int>(prediction == 1 && label == 1);
    int fn = static_cast<int>(prediction == 0 && label == 1);
    int tn = static_cast<int>(prediction == 0 && label == 0);
    int fp = static_cast<int>(prediction == 1 && label == 0);

    num_tp += tp;
    num_fn += fn;
    num_tn += tn;
    num_fp += fp;
  }

  BinaryAccuracyParameter_AccuracyMode mode =
      this->layer_param_.binary_accuracy_param().accuracy_mode();

  if (mode == BinaryAccuracyParameter_AccuracyMode_Accuracy) {
    accuracy = (num_tp + num_tn) / (num_tp + num_fn + num_tn + num_fp);
  }
  else if (mode == BinaryAccuracyParameter_AccuracyMode_Precision) {
    accuracy = (num_tp) / (num_tp + num_fp);
  }
  else if (mode == BinaryAccuracyParameter_AccuracyMode_Recall) {
    accuracy = (num_tp) / (num_tp + num_fn);
  }
  else if (mode == BinaryAccuracyParameter_AccuracyMode_F_measure) {
    Dtype precision = (num_tp) / (num_tp + num_fp);
    Dtype recall = (num_tp) / (num_tp + num_fn);
    accuracy = (2 * precision * recall) / (precision + recall);
  }
  else {
    LOG(ERROR) << "undefined BinaryAccuracyType";
  }

  // LOG(INFO) << "Accuracy: " << accuracy;
  top[0]->mutable_cpu_data()[0] = accuracy;
  // Accuracy layer should not be used as a loss function.
}

INSTANTIATE_CLASS(BinaryAccuracyLayer);
REGISTER_LAYER_CLASS(BIN_ACCURACY, BinaryAccuracyLayer);
}  // namespace caffe
