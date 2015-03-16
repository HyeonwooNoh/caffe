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
void RedAccuracyLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(bottom[0]);
  softmax_top_vec_.clear();
  softmax_top_vec_.push_back(&prob_);
  softmax_layer_->SetUp(softmax_bottom_vec_, softmax_top_vec_);

  top_k_ = this->layer_param_.red_accuracy_param().top_k();

  has_ignore_label_ =
    this->layer_param_.red_accuracy_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.red_accuracy_param().ignore_label();
  }

  // set redendant class num & class num
  red_cls_num_ = bottom[0]->channels();
  cls_num_ = red_cls_num_;
  if (this->layer_param_.red_softmax_loss_param().has_class_num()) {
    cls_num_ = this->layer_param_.red_softmax_loss_param().class_num();
  }
}

template <typename Dtype>
void RedAccuracyLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->num(), bottom[1]->num())
      << "The data and label should have the same number.";
  CHECK_LE(top_k_, bottom[0]->count() / bottom[0]->num())
      << "top_k must be less than or equal to the number of classes.";
  CHECK_EQ(bottom[1]->channels(), 1)
      << "Label data should have channel 1.";
  CHECK_EQ(bottom[0]->height(), bottom[1]->height())
      << "The data and label should have the same height.";
  CHECK_EQ(bottom[0]->width(), bottom[1]->width())
      << "the data and label should have the same width.";

  softmax_layer_->Reshape(softmax_bottom_vec_, softmax_top_vec_);

  top[0]->Reshape(1, 1, 1, 1);
}

template <typename Dtype>
void RedAccuracyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Dtype accuracy = 0;
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  int num = bottom[0]->num();
  int dim = bottom[0]->count() / bottom[0]->num();
  int spatial_dim = bottom[0]->height() * bottom[0]->width();
  int channels = bottom[0]->channels();
  int ignored_pixel_num = 0;
  vector<Dtype> maxval(top_k_+1);
  vector<int> max_id(top_k_+1);
  for (int i = 0; i < num; ++i) {
    for (int j = 0; j < spatial_dim; j++){
      const int label_value = static_cast<int>(bottom_label[i * spatial_dim + j]);
      if (has_ignore_label_ && label_value == ignore_label_) {
        ignored_pixel_num++;
        continue;
      }
      // Top-k accuracy
      std::vector<std::pair<Dtype, int> > bottom_data_vector;
      for (int k = 0; k < channels; ++k) {
        bottom_data_vector.push_back(
          std::make_pair(bottom_data[i * dim + k * spatial_dim + j], k));
      }
      std::partial_sort(
          bottom_data_vector.begin(), bottom_data_vector.begin() + top_k_,
          bottom_data_vector.end(), std::greater<std::pair<Dtype, int> >());
      // check if true label is in top k predictions
      for (int k = 0; k < top_k_; k++) {
        if (bottom_data_vector[k].second == label_value) {
          ++accuracy;
          break;
        }
      }
    }
  }
  top[0]->mutable_cpu_data()[0] = accuracy / (num * spatial_dim - ignored_pixel_num);
  // Accuracy layer should not be used as a loss function.
}

INSTANTIATE_CLASS(RedAccuracyLayer);
REGISTER_LAYER_CLASS(RED_ACCURACY, RedAccuracyLayer);
}  // namespace caffe
