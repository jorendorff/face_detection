/*
 *
 * This file is part of the open-source SeetaFace engine, which includes three modules:
 * SeetaFace Detection, SeetaFace Alignment, and SeetaFace Identification.
 *
 * This file is part of the SeetaFace Detection module, containing codes implementing the
 * face detection method described in the following paper:
 *
 *
 *   Funnel-structured cascade for multi-view face detection with alignment awareness,
 *   Shuzhe Wu, Meina Kan, Zhenliang He, Shiguang Shan, Xilin Chen.
 *   In Neurocomputing (under review)
 *
 *
 * Copyright (C) 2016, Visual Information Processing and Learning (VIPL) group,
 * Institute of Computing Technology, Chinese Academy of Sciences, Beijing, China.
 *
 * The codes are mainly developed by Shuzhe Wu (a Ph.D supervised by Prof. Shiguang Shan)
 *
 * As an open-source face recognition engine: you can redistribute SeetaFace source codes
 * and/or modify it under the terms of the BSD 2-Clause License.
 *
 * You should have received a copy of the BSD 2-Clause License along with the software.
 * If not, see < https://opensource.org/licenses/BSD-2-Clause>.
 *
 * Contact Info: you can send an email to SeetaFace@vipl.ict.ac.cn for any problems.
 *
 * Note: the above information must be kept whenever or wherever the codes are used.
 *
 */

#ifndef SEETA_FACE_DETECTION_H_
#define SEETA_FACE_DETECTION_H_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iosfwd>
#include <istream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#ifdef USE_SSE
#include <immintrin.h>
#endif

#ifdef USE_OPENMP
#include <omp.h>

#define SEETA_NUM_THREADS 4
#endif


// include/common.h

#if defined _WIN32
#ifdef SEETA_EXPORTS
#define SEETA_API __declspec(dllexport)
#else
#define SEETA_API __declspec(dllimport)
#endif

#else
#define SEETA_API
#endif

#define DISABLE_COPY_AND_ASSIGN(classname) \
 private: \
  classname(const classname&); \
  classname& operator=(const classname&)

namespace seeta {

typedef struct ImageData {
  ImageData() {
    data = nullptr;
    width = 0;
    height = 0;
    num_channels = 0;
  }

  ImageData(int32_t img_width, int32_t img_height,
      int32_t img_num_channels = 1) {
    data = nullptr;
    width = img_width;
    height = img_height;
    num_channels = img_num_channels;
  }

  uint8_t* data;
  int32_t width;
  int32_t height;
  int32_t num_channels;
} ImageData;

typedef struct Rect {
  int32_t x;
  int32_t y;
  int32_t width;
  int32_t height;
} Rect;

typedef struct FaceInfo {
  seeta::Rect bbox;

  double roll;
  double pitch;
  double yaw;

  double score; /**< Larger score should mean higher confidence. */
} FaceInfo;

  typedef struct {
    double x;
    double y;
  } FacialLandmark;


// include/feature_map.h

namespace fd {

class FeatureMap {
 public:
  FeatureMap()
      : width_(0), height_(0) {
    roi_.x = 0;
    roi_.y = 0;
    roi_.width = 0;
    roi_.height = 0;
  }

  virtual ~FeatureMap() {}

  virtual void Compute(const uint8_t* input, int32_t width, int32_t height) = 0;

  inline virtual void SetROI(const seeta::Rect & roi) {
    roi_ = roi;
  }

 protected:
  int32_t width_;
  int32_t height_;

  seeta::Rect roi_;
};


// include/classifier.h

enum ClassifierType {
    LAB_Boosted_Classifier,
    SURF_MLP
};

class Classifier {
 public:
  Classifier() {}
  virtual ~Classifier() {}

  virtual void SetFeatureMap(seeta::fd::FeatureMap* feat_map) = 0;
  virtual bool Classify(float* score = nullptr, float* outputs = nullptr) = 0;

  virtual seeta::fd::ClassifierType type() = 0;

  DISABLE_COPY_AND_ASSIGN(Classifier);
};


// include/util/image_pyramid.h

static void ResizeImage(const seeta::ImageData & src, seeta::ImageData* dest) {
  int32_t src_width = src.width;
  int32_t src_height = src.height;
  int32_t dest_width = dest->width;
  int32_t dest_height = dest->height;
  if (src_width == dest_width && src_height == dest_height) {
    std::memcpy(dest->data, src.data, src_width * src_height * sizeof(uint8_t));
    return;
  }

  double lf_x_scl = static_cast<double>(src_width) / dest_width;
  double lf_y_Scl = static_cast<double>(src_height) / dest_height;
  const uint8_t* src_data = src.data;
  uint8_t* dest_data = dest->data;

#pragma omp parallel num_threads(SEETA_NUM_THREADS)
  {
#pragma omp for nowait
    for (int32_t y = 0; y < dest_height; y++) {
      for (int32_t x = 0; x < dest_width; x++) {
        double lf_x_s = lf_x_scl * x;
        double lf_y_s = lf_y_Scl * y;

        int32_t n_x_s = static_cast<int>(lf_x_s);
        n_x_s = (n_x_s <= (src_width - 2) ? n_x_s : (src_width - 2));
        int32_t n_y_s = static_cast<int>(lf_y_s);
        n_y_s = (n_y_s <= (src_height - 2) ? n_y_s : (src_height - 2));

        double lf_weight_x = lf_x_s - n_x_s;
        double lf_weight_y = lf_y_s - n_y_s;

        double dest_val = (1 - lf_weight_y) * ((1 - lf_weight_x) *
          src_data[n_y_s * src_width + n_x_s] +
          lf_weight_x * src_data[n_y_s * src_width + n_x_s + 1]) +
          lf_weight_y * ((1 - lf_weight_x) * src_data[(n_y_s + 1) * src_width + n_x_s] +
          lf_weight_x * src_data[(n_y_s + 1) * src_width + n_x_s + 1]);

        dest_data[y * dest_width + x] = static_cast<uint8_t>(dest_val);
      }
    }
  }
}

class ImagePyramid {
 public:
  ImagePyramid()
      : max_scale_(1.0f), min_scale_(1.0f),
        scale_factor_(1.0f), scale_step_(0.8f),
        width1x_(0), height1x_(0),
        width_scaled_(0), height_scaled_(0),
        buf_img_width_(2), buf_img_height_(2),
        buf_scaled_width_(2), buf_scaled_height_(2) {
    buf_img_ = new uint8_t[buf_img_width_ * buf_img_height_];
    buf_img_scaled_ = new uint8_t[buf_scaled_width_ * buf_scaled_height_];
  }

  ~ImagePyramid() {
    delete[] buf_img_;
    buf_img_ = nullptr;

    buf_img_width_ = 0;
    buf_img_height_ = 0;

    delete[] buf_img_scaled_;
    buf_img_scaled_ = nullptr;

    buf_scaled_width_ = 0;
    buf_scaled_height_ = 0;

    img_scaled_.data = nullptr;
    img_scaled_.width = 0;
    img_scaled_.height = 0;
  }

  inline void SetScaleStep(float step) {
    if (step > 0.0f && step <= 1.0f)
      scale_step_ = step;
  }

  inline void SetMinScale(float min_scale) {
    min_scale_ = min_scale;
  }

  inline void SetMaxScale(float max_scale) {
    max_scale_ = max_scale;
    scale_factor_ = max_scale;
    UpdateBufScaled();
  }

  void SetImage1x(const uint8_t* img_data, int32_t width, int32_t height);

  inline float min_scale() const { return min_scale_; }
  inline float max_scale() const { return max_scale_; }

  inline seeta::ImageData image1x() {
    seeta::ImageData img(width1x_, height1x_, 1);
    img.data = buf_img_;
    return img;
  }

  const seeta::ImageData* GetNextScaleImage(float* scale_factor = nullptr);

 private:
  void UpdateBufScaled();

  float max_scale_;
  float min_scale_;

  float scale_factor_;
  float scale_step_;

  int32_t width1x_;
  int32_t height1x_;

  int32_t width_scaled_;
  int32_t height_scaled_;

  uint8_t* buf_img_;
  int32_t buf_img_width_;
  int32_t buf_img_height_;

  uint8_t* buf_img_scaled_;
  int32_t buf_scaled_width_;
  int32_t buf_scaled_height_;

  seeta::ImageData img_scaled_;
};


// include/detector.h

class Detector {
 public:
  Detector() {}
  virtual ~Detector() {}

  virtual bool LoadModel(const std::string & model_path) = 0;
  virtual std::vector<seeta::FaceInfo> Detect(seeta::fd::ImagePyramid* img_pyramid) = 0;

  virtual void SetWindowSize(int32_t size) {}
  virtual void SetSlideWindowStep(int32_t step_x, int32_t step_y) {}

  DISABLE_COPY_AND_ASSIGN(Detector);
};


// include/model_reader.h

class ModelReader {
 public:
  ModelReader() {}
  virtual ~ModelReader() {}

  virtual bool Read(std::istream* input, seeta::fd::Classifier* model) = 0;

  DISABLE_COPY_AND_ASSIGN(ModelReader);
};


// include/fust.h

class FuStDetector : public Detector {
 public:
  FuStDetector()
      : wnd_size_(40), slide_wnd_step_x_(4), slide_wnd_step_y_(4),
        num_hierarchy_(0) {
    wnd_data_buf_.resize(wnd_size_ * wnd_size_);
    wnd_data_.resize(wnd_size_ * wnd_size_);
  }

  ~FuStDetector() {}

  virtual bool LoadModel(const std::string & model_path);
  virtual std::vector<seeta::FaceInfo> Detect(seeta::fd::ImagePyramid* img_pyramid);

  inline virtual void SetWindowSize(int32_t size) {
    if (size >= 20)
      wnd_size_ = size;
  }

  inline virtual void SetSlideWindowStep(int32_t step_x, int32_t step_y) {
    if (step_x > 0)
      slide_wnd_step_x_ = step_x;
    if (step_y > 0)
      slide_wnd_step_y_ = step_y;
  }

 private:
  std::shared_ptr<seeta::fd::ModelReader> CreateModelReader(seeta::fd::ClassifierType type);
  std::shared_ptr<seeta::fd::Classifier> CreateClassifier(seeta::fd::ClassifierType type);
  std::shared_ptr<seeta::fd::FeatureMap> CreateFeatureMap(seeta::fd::ClassifierType type);

  void GetWindowData(const seeta::ImageData & img, const seeta::Rect & wnd);

  int32_t wnd_size_;
  int32_t slide_wnd_step_x_;
  int32_t slide_wnd_step_y_;

  int32_t num_hierarchy_;
  std::vector<int32_t> hierarchy_size_;
  std::vector<int32_t> num_stage_;
  std::vector<std::vector<int32_t> > wnd_src_id_;

  std::vector<uint8_t> wnd_data_buf_;
  std::vector<uint8_t> wnd_data_;

  std::vector<std::shared_ptr<seeta::fd::Classifier> > model_;
  std::vector<std::shared_ptr<seeta::fd::FeatureMap> > feat_map_;
  std::map<seeta::fd::ClassifierType, int32_t> cls2feat_idx_;

  DISABLE_COPY_AND_ASSIGN(FuStDetector);
};


// include/util/nms.h

void NonMaximumSuppression(std::vector<seeta::FaceInfo>* bboxes,
  std::vector<seeta::FaceInfo>* bboxes_nms, float iou_thresh = 0.8f);


// include/util/math_func.h

class MathFunction {
 public:
  static inline void UInt8ToInt32(const uint8_t* src, int32_t* dest,
      int32_t len) {
    for (int32_t i = 0; i < len; i++)
      *(dest++) = static_cast<int32_t>(*(src++));
  }

  static inline void VectorAdd(const int32_t* x, const int32_t* y, int32_t* z,
      int32_t len) {
    int32_t i;
#ifdef USE_SSE
    __m128i x1;
    __m128i y1;
    const __m128i* x2 = reinterpret_cast<const __m128i*>(x);
    const __m128i* y2 = reinterpret_cast<const __m128i*>(y);
    __m128i* z2 = reinterpret_cast<__m128i*>(z);

    for (i = 0; i < len - 4; i += 4) {
      x1 = _mm_loadu_si128(x2++);
      y1 = _mm_loadu_si128(y2++);
      _mm_storeu_si128(z2++, _mm_add_epi32(x1, y1));
    }
    for (; i < len; i++)
      *(z + i) = (*(x + i)) + (*(y + i));
#else
    for (i = 0; i < len; i++)
      *(z + i) = (*(x + i)) + (*(y + i));
#endif
  }

  static inline void VectorSub(const int32_t* x, const int32_t* y, int32_t* z,
      int32_t len) {
    int32_t i;
#ifdef USE_SSE
    __m128i x1;
    __m128i y1;
    const __m128i* x2 = reinterpret_cast<const __m128i*>(x);
    const __m128i* y2 = reinterpret_cast<const __m128i*>(y);
    __m128i* z2 = reinterpret_cast<__m128i*>(z);

    for (i = 0; i < len - 4; i += 4) {
      x1 = _mm_loadu_si128(x2++);
      y1 = _mm_loadu_si128(y2++);

      _mm_storeu_si128(z2++, _mm_sub_epi32(x1, y1));
    }
    for (; i < len; i++)
      *(z + i) = (*(x + i)) - (*(y + i));
#else
    for (i = 0; i < len; i++)
      *(z + i) = (*(x + i)) - (*(y + i));
#endif
  }

  static inline void VectorAbs(const int32_t* src, int32_t* dest, int32_t len) {
    int32_t i;
#ifdef USE_SSE
    __m128i val;
    __m128i val_abs;
    const __m128i* x = reinterpret_cast<const __m128i*>(src);
    __m128i* y = reinterpret_cast<__m128i*>(dest);

    for (i = 0; i < len - 4; i += 4) {
      val = _mm_loadu_si128(x++);
      val_abs = _mm_abs_epi32(val);
      _mm_storeu_si128(y++, val_abs);
    }
    for (; i < len; i++)
      dest[i] = (src[i] >= 0 ? src[i] : -src[i]);
#else
    for (i = 0; i < len; i++)
      dest[i] = (src[i] >= 0 ? src[i] : -src[i]);
#endif
  }

  static inline void Square(const int32_t* src, uint32_t* dest, int32_t len) {
    int32_t i;
#ifdef USE_SSE
    __m128i x1;
    const __m128i* x2 = reinterpret_cast<const __m128i*>(src);
    __m128i* y2 = reinterpret_cast<__m128i*>(dest);

    for (i = 0; i < len - 4; i += 4) {
      x1 = _mm_loadu_si128(x2++);
      _mm_storeu_si128(y2++, _mm_mullo_epi32(x1, x1));
    }
    for (; i < len; i++)
      *(dest + i) = (*(src + i)) * (*(src + i));
#else
    for (i = 0; i < len; i++)
      *(dest + i) = (*(src + i)) * (*(src + i));
#endif
  }

  static inline float VectorInnerProduct(const float* x, const float* y,
      int32_t len) {
    float prod = 0;
    int32_t i;
#ifdef USE_SSE
    __m128 x1;
    __m128 y1;
    __m128 z1 = _mm_setzero_ps();
    float buf[4];

    for (i = 0; i < len - 4; i += 4) {
      x1 = _mm_loadu_ps(x + i);
      y1 = _mm_loadu_ps(y + i);
      z1 = _mm_add_ps(z1, _mm_mul_ps(x1, y1));
    }
    _mm_storeu_ps(&buf[0], z1);
    prod = buf[0] + buf[1] + buf[2] + buf[3];
    for (; i < len; i++)
      prod += x[i] * y[i];
#else
    for (i = 0; i < len; i++)
        prod += x[i] * y[i];
#endif
    return prod;
  }
};


// include/feat/lab_feature_map.h

/** @struct LABFeature
 *  @brief Locally Assembled Binary (LAB) feature.
 *
 *  It is parameterized by the coordinates of top left corner.
 */
typedef struct LABFeature {
  int32_t x;
  int32_t y;
} LABFeature;

class LABFeatureMap : public seeta::fd::FeatureMap {
 public:
  LABFeatureMap() : rect_width_(3), rect_height_(3), num_rect_(3) {}
  virtual ~LABFeatureMap() {}

  virtual void Compute(const uint8_t* input, int32_t width, int32_t height);

  inline uint8_t GetFeatureVal(int32_t offset_x, int32_t offset_y) const {
    return feat_map_[(roi_.y + offset_y) * width_ + roi_.x + offset_x];
  }

  float GetStdDev() const;

 private:
  void Reshape(int32_t width, int32_t height);
  void ComputeIntegralImages(const uint8_t* input);
  void ComputeRectSum();
  void ComputeFeatureMap();

  template<typename Int32Type>
  inline void Integral(Int32Type* data) {
    const Int32Type* src = data;
    Int32Type* dest = data;
    const Int32Type* dest_above = dest;

    *dest = *(src++);
    for (int32_t c = 1; c < width_; c++, src++, dest++)
      *(dest + 1) = (*dest) + (*src);
    dest++;
    for (int32_t r = 1; r < height_; r++) {
      for (int32_t c = 0, s = 0; c < width_; c++, src++, dest++, dest_above++) {
        s += (*src);
        *dest = *dest_above + s;
      }
    }
  }

  const int32_t rect_width_;
  const int32_t rect_height_;
  const int32_t num_rect_;

  std::vector<uint8_t> feat_map_;
  std::vector<int32_t> rect_sum_;
  std::vector<int32_t> int_img_;
  std::vector<uint32_t> square_int_img_;
};


// include/feat/surf_feature_map.h

typedef struct SURFFeature {
  seeta::Rect patch;
  int32_t num_cell_per_row;
  int32_t num_cell_per_col;
} SURFFeature;

class SURFFeaturePool {
 public:
  SURFFeaturePool()
      : sample_width_(40), sample_height_(40),
        patch_move_step_x_(16), patch_move_step_y_(16), patch_size_inc_step_(1),
        patch_min_width_(16), patch_min_height_(16) {}

  ~SURFFeaturePool() {}

  void Create();
  void AddPatchFormat(int32_t width, int32_t height, int32_t num_cell_per_row,
      int32_t num_cell_per_col);

  inline bool empty() const { return pool_.empty(); }
  inline std::size_t size() const { return pool_.size(); }

  inline std::vector<SURFFeature>::const_iterator begin() const {
    return pool_.begin();
  }

  inline std::vector<SURFFeature>::const_iterator end() const {
    return pool_.end();
  }

  inline const SURFFeature & operator[](std::size_t idx) const {
    return pool_[idx];
  }

 private:
  void AddAllFeaturesToPool(int32_t width, int32_t height,
      int32_t num_cell_per_row, int32_t num_cell_per_col);

  typedef struct SURFPatchFormat {
    /**< aspect ratio, s.t. GCD(width, height) = 1 */
    int32_t width;
    int32_t height;

    /**< cell partition */
    int32_t num_cell_per_row;
    int32_t num_cell_per_col;
  } SURFPatchFormat;

  int32_t sample_width_;
  int32_t sample_height_;
  int32_t patch_move_step_x_;
  int32_t patch_move_step_y_;
  int32_t patch_size_inc_step_; /**< incremental step of patch width and */
                                /**< height when build feature pool      */
  int32_t patch_min_width_;
  int32_t patch_min_height_;

  std::vector<SURFFeature> pool_;
  std::vector<SURFPatchFormat> format_;
};

class SURFFeatureMap : public FeatureMap {
 public:
  SURFFeatureMap() : buf_valid_reset_(false) { InitFeaturePool(); }
  virtual ~SURFFeatureMap() {}

  virtual void Compute(const uint8_t* input, int32_t width, int32_t height);

  inline virtual void SetROI(const seeta::Rect & roi) {
    roi_ = roi;
    if (buf_valid_reset_) {
      std::memset(buf_valid_.data(), 0, buf_valid_.size() * sizeof(int32_t));
      buf_valid_reset_ = false;
    }
  }

  inline int32_t GetFeatureVectorDim(int32_t feat_id) const {
    return (feat_pool_[feat_id].num_cell_per_col *
      feat_pool_[feat_id].num_cell_per_row * kNumIntChannel);
  }

  void GetFeatureVector(int32_t featID, float* featVec);

 private:
  void InitFeaturePool();
  void Reshape(int32_t width, int32_t height);

  void ComputeGradientImages(const uint8_t* input);
  void ComputeGradX(const int32_t* input);
  void ComputeGradY(const int32_t* input);
  void ComputeIntegralImages();
  void Integral();
  void MaskIntegralChannel();

  inline void FillIntegralChannel(const int32_t* src, int32_t ch) {
    int32_t* dest = int_img_.data() + ch;
    int32_t len = width_ * height_;
    for (int32_t i = 0; i < len; i++) {
      *dest = *src;
      *(dest + 2) = *src;
      dest += kNumIntChannel;
      src++;
    }
  }

  void ComputeFeatureVector(const SURFFeature & feat, int32_t* feat_vec);
  void NormalizeFeatureVectorL2(const int32_t* feat_vec, float* feat_vec_normed,
    int32_t len) const;

  /**
   * Number of channels should be divisible by 4.
   */
  void VectorCumAdd(int32_t* x, int32_t len, int32_t num_channel);

  static const int32_t kNumIntChannel = 8;

  bool buf_valid_reset_;

  std::vector<int32_t> grad_x_;
  std::vector<int32_t> grad_y_;
  std::vector<int32_t> int_img_;
  std::vector<int32_t> img_buf_;
  std::vector<std::vector<int32_t> > feat_vec_buf_;
  std::vector<std::vector<float> > feat_vec_normed_buf_;
  std::vector<int32_t> buf_valid_;

  seeta::fd::SURFFeaturePool feat_pool_;
};


// include/classifier/mlp.h

class MLPLayer {
 public:
  explicit MLPLayer(int32_t act_func_type = 1)
      : input_dim_(0), output_dim_(0), act_func_type_(act_func_type) {}
  ~MLPLayer() {}

  void Compute(const float* input, float* output);

  inline int32_t GetInputDim() const { return input_dim_; }
  inline int32_t GetOutputDim() const { return output_dim_; }

  inline void SetSize(int32_t inputDim, int32_t outputDim) {
    if (inputDim <= 0 || outputDim <= 0) {
      return;  // @todo handle the errors!!!
    }
    input_dim_ = inputDim;
    output_dim_ = outputDim;
    weights_.resize(inputDim * outputDim);
    bias_.resize(outputDim);
  }

  inline void SetWeights(const float* weights, int32_t len) {
    if (weights == nullptr || len != input_dim_ * output_dim_) {
      return;  // @todo handle the errors!!!
    }
    std::copy(weights, weights + input_dim_ * output_dim_, weights_.begin());
  }

  inline void SetBias(const float* bias, int32_t len) {
    if (bias == nullptr || len != output_dim_) {
      return;  // @todo handle the errors!!!
    }
    std::copy(bias, bias + output_dim_, bias_.begin());
  }

 private:
  inline float Sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(x));
  }

  inline float ReLU(float x) {
    return (x > 0.0f ? x : 0.0f);
  }

 private:
  int32_t act_func_type_;
  int32_t input_dim_;
  int32_t output_dim_;
  std::vector<float> weights_;
  std::vector<float> bias_;
};


class MLP {
 public:
  MLP() {}
  ~MLP() {}

  void Compute(const float* input, float* output);

  inline int32_t GetInputDim() const {
    return layers_[0]->GetInputDim();
  }

  inline int32_t GetOutputDim() const {
    return layers_.back()->GetOutputDim();
  }

  inline int32_t GetLayerNum() const {
    return static_cast<int32_t>(layers_.size());
  }

  void AddLayer(int32_t inputDim, int32_t outputDim, const float* weights,
      const float* bias, bool is_output = false);

 private:
  std::vector<std::shared_ptr<seeta::fd::MLPLayer> > layers_;
  std::vector<float> layer_buf_[2];
};


// include/classifier/surf_mlp.h

class SURFMLP : public Classifier {
 public:
  SURFMLP() : Classifier(), model_(new seeta::fd::MLP()) {}
  virtual ~SURFMLP() {}

  virtual bool Classify(float* score = nullptr, float* outputs = nullptr);

  inline virtual void SetFeatureMap(seeta::fd::FeatureMap* feat_map) {
    feat_map_ = dynamic_cast<seeta::fd::SURFFeatureMap*>(feat_map);
  }

  inline virtual seeta::fd::ClassifierType type() {
    return seeta::fd::ClassifierType::SURF_MLP;
  }

  void AddFeatureByID(int32_t feat_id);
  void AddLayer(int32_t input_dim, int32_t output_dim, const float* weights,
    const float* bias, bool is_output = false);

  inline void SetThreshold(float thresh) { thresh_ = thresh; }

 private:
  std::vector<int32_t> feat_id_;
  std::vector<float> input_buf_;
  std::vector<float> output_buf_;

  std::shared_ptr<seeta::fd::MLP> model_;
  float thresh_;
  seeta::fd::SURFFeatureMap* feat_map_;
};


// include/classifier/lab_boosted_classifier.h

/**
 * @class LABBaseClassifier
 * @brief Base classifier using LAB feature.
 */
class LABBaseClassifier {
 public:
  LABBaseClassifier()
    : num_bin_(255), thresh_(0.0f) {
    weights_.resize(num_bin_ + 1);
  }

  ~LABBaseClassifier() {}

  void SetWeights(const float* weights, int32_t num_bin);

  inline void SetThreshold(float thresh) { thresh_ = thresh; }

  inline int32_t num_bin() const { return num_bin_; }
  inline float weights(int32_t val) const { return weights_[val]; }
  inline float threshold() const { return thresh_; }

 private:
  int32_t num_bin_;

  std::vector<float> weights_;
  float thresh_;
};

/**
 * @class LABBoostedClassifier
 * @Brief A strong classifier constructed from base classifiers using LAB features.
 */
class LABBoostedClassifier : public Classifier {
 public:
  LABBoostedClassifier() : use_std_dev_(true) {}
  virtual ~LABBoostedClassifier() {}

  virtual bool Classify(float* score = nullptr, float* outputs = nullptr);

  inline virtual seeta::fd::ClassifierType type() {
    return seeta::fd::ClassifierType::LAB_Boosted_Classifier;
  }

  void AddFeature(int32_t x, int32_t y);
  void AddBaseClassifier(const float* weights, int32_t num_bin, float thresh);

  inline virtual void SetFeatureMap(seeta::fd::FeatureMap* featMap) {
    feat_map_ = dynamic_cast<seeta::fd::LABFeatureMap*>(featMap);
  }

  inline void SetUseStdDev(bool useStdDev) { use_std_dev_ = useStdDev; }

 private:
  static const int32_t kFeatGroupSize = 10;
  const float kStdDevThresh = 10.0f;

  std::vector<seeta::fd::LABFeature> feat_;
  std::vector<std::shared_ptr<seeta::fd::LABBaseClassifier> > base_classifiers_;
  seeta::fd::LABFeatureMap* feat_map_;
  bool use_std_dev_;
};

}  // namespace fd


// include/face_detection.h

class FaceDetection {
 public:
  SEETA_API explicit FaceDetection(const char* model_path);
  SEETA_API ~FaceDetection();

  /**
   * @brief Detect faces on input image.
   *
   * (1) The input image should be gray-scale, i.e. `num_channels` set to 1.
   * (2) Currently this function does not give the Euler angles, which are
   *     left with invalid values.
   */
  SEETA_API std::vector<seeta::FaceInfo> Detect(const seeta::ImageData & img);

  /**
   * @brief Set the minimum size of faces to detect.
   *
   * The minimum size is constrained as no smaller than 20. Invalid values will
   * be ignored.
   */
  SEETA_API void SetMinFaceSize(int32_t size);

  /**
   * @brief Set the maximum size of faces to detect.
   *
   * The maximum face size actually used is computed as the minimum among: user
   * specified size, image width, image height.
   */
  SEETA_API void SetMaxFaceSize(int32_t size);

  /**
   * @brief Set the factor between adjacent scales of image pyramid.
   *
   * The value of the factor lies in (0, 1). For example, when it is set as 0.5,
   * an input image of size w x h will be resized to 0.5w x 0.5h, 0.25w x 0.25h,
   * 0.125w x 0.125h, etc. Invalid values will be ignored.
   */
  SEETA_API void SetImagePyramidScaleFactor(float factor);

  /**
   * @brief Set the sliding window step in horizontal and vertical directions.
   *
   * The steps should take positive values, and invalid ones will be ignored.
   * Usually a step of 4 is a reasonable choice.
   */
  SEETA_API void SetWindowStep(int32_t step_x, int32_t step_y);

  /**
   * @brief Set the score thresh of detected faces.
   *
   * Detections with scores smaller than the threshold will not be returned.
   * Typical threshold values include 0.95, 2.8, 4.5. One can adjust the
   * threshold based on his or her own test set.
   */
  SEETA_API void SetScoreThresh(float thresh);

  DISABLE_COPY_AND_ASSIGN(FaceDetection);

 private:
  class Impl;
  Impl* impl_;
};


// include/io/lab_boost_model_reader.h

namespace fd {

class LABBoostModelReader : public ModelReader {
 public:
  LABBoostModelReader() : ModelReader() {}
  virtual ~LABBoostModelReader() {}

  virtual bool Read(std::istream* input, seeta::fd::Classifier* model);

 private:
  bool ReadFeatureParam(std::istream* input,
    seeta::fd::LABBoostedClassifier* model);
  bool ReadBaseClassifierParam(std::istream* input,
    seeta::fd::LABBoostedClassifier* model);

  int32_t num_bin_;
  int32_t num_base_classifer_;
};


// include/io/surf_mlp_model_reader.h

class SURFMLPModelReader : public ModelReader {
 public:
  SURFMLPModelReader() {}
  virtual ~SURFMLPModelReader() {}

  virtual bool Read(std::istream* input, seeta::fd::Classifier* model);

 private:
  std::vector<int32_t> feat_id_buf_;
  std::vector<float> weights_buf_;
  std::vector<float> bias_buf_;
};


}  // namespace fd
}  // namespace seeta

#endif  // SEETA_FACE_DETECTION_H_
