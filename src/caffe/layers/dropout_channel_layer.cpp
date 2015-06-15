// TODO (sergeyk): effect should not be dependent on phase. wasted memcpy.

#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void DropoutChannelLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::LayerSetUp(bottom, top);
  threshold_ = this->layer_param_.dropout_param().dropout_ratio();
  DCHECK(threshold_ > 0.);
  DCHECK(threshold_ < 1.);
  scale_ = 1. / (1. - threshold_);
  uint_thres_ = static_cast<unsigned int>(UINT_MAX * threshold_);

  // Figure out the dimensions
  N_ = bottom[0]->num();
  C_ = bottom[0]->channels();
  H_ = bottom[0]->height();
  W_ = bottom[0]->width();
}

template <typename Dtype>
void DropoutChannelLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::Reshape(bottom, top);
  // Set up the cache for random number generation
  rand_mat_.Reshape(bottom[0]->num(), bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());
  rand_vec_.Reshape(bottom[0]->num(), bottom[0]->channels(),
      1, 1);

  // fill spatial multiplier
  spatial_sum_multiplier_.Reshape(1, 1, bottom[0]->height(), bottom[0]->width());
  Dtype* spatial_multipl_data = spatial_sum_multiplier_.mutable_cpu_data();
  caffe_set(spatial_sum_multiplier_.count(), Dtype(1),
      spatial_multipl_data);
  caffe_set(spatial_sum_multiplier_.count(), Dtype(0),
      spatial_sum_multiplier_.mutable_cpu_diff());
}

template <typename Dtype>
void DropoutChannelLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const Dtype* const_vec = rand_vec_.cpu_data();
  Dtype* vec = rand_vec_.mutable_cpu_data();
  Dtype* mask = rand_mat_.mutable_cpu_data();
  const int count = bottom[0]->count();
  if (Caffe::phase() == Caffe::TRAIN) {
    // Create random numbers
    caffe_rng_uniform(rand_vec_.count(), Dtype(0), Dtype(1), vec);
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, 
                          N_ * C_, H_ * W_, 1, Dtype(1),
                          const_vec, spatial_sum_multiplier_.cpu_data(),
                          Dtype(0), mask); 

    for (int i = 0; i < count; ++i) {
      top_data[i] = bottom_data[i] * (mask[i] > threshold_) * scale_;
    }
  } else {
    caffe_copy(bottom[0]->count(), bottom_data, top_data);
  }
}

template <typename Dtype>
void DropoutChannelLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    if (Caffe::phase() == Caffe::TRAIN) {
      const Dtype* mask = rand_mat_.cpu_data();
      const int count = bottom[0]->count();
      for (int i = 0; i < count; ++i) {
        bottom_diff[i] = top_diff[i] * (mask[i] > threshold_) * scale_;
      }
    } else {
      caffe_copy(top[0]->count(), top_diff, bottom_diff);
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(DropoutChannelLayer);
#endif

INSTANTIATE_CLASS(DropoutChannelLayer);
REGISTER_LAYER_CLASS(DROPOUT_CHANNEL, DropoutChannelLayer);
}  // namespace caffe
