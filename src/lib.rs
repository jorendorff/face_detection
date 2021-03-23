extern crate ordered_float;

use ordered_float::OrderedFloat;

#[derive(Clone)]
pub struct Rect {
    x: i32,
    y: i32,
    width: i32,
    height: i32,
}

#[derive(Clone)]
pub struct FaceInfo {
    bbox: Rect,
    roll: f64,
    pitch: f64,
    yaw: f64,

    /// Larger score means higher confidence.
    score: f64,
}

struct ImageData<'buf> {
    data: &'buf [u8],
    width: usize,
    height: usize,
    num_channels: usize,
}

/*
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
*/

pub extern "C" fn non_maximum_suppression(
    mut bboxes: Vec<FaceInfo>,
    iou_thresh: f32,
) -> Vec<FaceInfo> {
    let mut bboxes_nms = vec![];
    bboxes.sort_by_key(|fi| OrderedFloat(fi.score));

    let num_bbox = bboxes.len();
    let mut mask_merged = vec![false; num_bbox];

    for select_idx in 0 .. num_bbox {
        if mask_merged[select_idx] {
            continue;
        }

        bboxes_nms.push(bboxes[select_idx].clone());
        mask_merged[select_idx] = true;

        let select_bbox = &bboxes[select_idx].bbox;
        let area1 = (select_bbox.width * select_bbox.height) as f32;
        let x1 = select_bbox.x as f32;
        let y1 = select_bbox.y as f32;
        let x2 = (select_bbox.x + select_bbox.width - 1) as f32;
        let y2 = (select_bbox.y + select_bbox.height - 1) as f32;

        for i in select_idx + 1 .. num_bbox {
            if mask_merged[i] {
                continue;
            }

            let bbox_i = &bboxes[i].bbox;
            let x = f32::max(x1, bbox_i.x as f32);
            let y = f32::max(y1, bbox_i.y as f32);
            let w = f32::min(x2, (bbox_i.x + bbox_i.width - 1) as f32) - x + 1.0;
            let h = f32::min(y2, (bbox_i.y + bbox_i.height - 1) as f32) - y + 1.0;
            if w <= 0.0 || h <= 0.0 {
                // The boxes do not overlap.
                continue;
            }

            let area2 = (bbox_i.width * bbox_i.height) as f32;
            let area_intersect = w * h;
            let area_union = area1 + area2 - area_intersect;
            if area_intersect as f32 / area_union > iou_thresh {
                mask_merged[i] = true;
                bboxes_nms.last_mut().unwrap().score += bboxes[i].score;
            }
        }
    }

    return bboxes_nms;
}


// include/util/image_pyramid.h

/*
fn resize_image<'a, 'b>(src: &ImageData<'a>, dest: &mut ImageData<'b>) {
    let src_width = src.width;
    let src_height = src.height;
    let dest_width = dest.width;
    let dest_height = dest.height;
    if src_width == dest_width && src_height == dest_height {
        std::memcpy(dest->data, src.data, src_width * src_height * sizeof(uint8_t));
        return;
    }

    let lf_x_scl = src_width as f64 / dest_width as f64;
    let lf_y_scl = src_height as f64 / dest_height as f64;
    let src_data = src.data;
    let dest_data = dest.data;

#pragma omp parallel num_threads(SEETA_NUM_THREADS)
    {
        #pragma omp for nowait
        for y in 0..dest_height {
            for x in 0..dest_width {
                let lf_x_s = lf_x_scl * x as f64;
                let lf_y_s = lf_y_scl * y as f64;

                let n_x_s = lf_x_s as i32; // XXX needs saturating_casts
                n_x_s = if n_x_s <= (src_width - 2) { n_x_s } else { src_width - 2 };
                let n_y_s = lf_y_s as i32;
                n_y_s = if n_y_s <= (src_height - 2) { n_y_s } else { src_height - 2 };

                let lf_weight_x = lf_x_s - n_x_s;
                let lf_weight_y = lf_y_s - n_y_s;

                let dest_val = (1 - lf_weight_y) * ((1 - lf_weight_x) *
                                                       src_data[n_y_s * src_width + n_x_s] +
                                                       lf_weight_x * src_data[n_y_s * src_width + n_x_s + 1]) +
                    lf_weight_y * ((1 - lf_weight_x) * src_data[(n_y_s + 1) * src_width + n_x_s] +
                                   lf_weight_x * src_data[(n_y_s + 1) * src_width + n_x_s + 1]);

                dest_data[y * dest_width + x] = static_cast<uint8_t>(dest_val);
            }
        }
    }
}

struct ImagePyramid {
    max_scale: f32,
    min_scale: f32,

    scale_factor: f32,
    scale_step: f32,

    width1x: usize,
    height1x: usize,

    width_scaled: usize,
    height_scaled: usize,

    buf_img: Vec<u8>,
    buf_img_width: usize,
    buf_img_height: usize,

    buf_img_scaled: Vec<u8>,
    buf_scaled_width: usize,
    buf_scaled_height: usize,

    img_scaled: ImageData,
}

impl ImagePyramid {
    pub fn new() -> ImagePyramid {
        const WIDTH: usize = 2;
        const HEIGHT: usize = 2;
        ImagePyramid {
            max_scale_: 1.0f32,
            min_scale_: 1.0f32,
            scale_factor_: 1.0f32,
            scale_step_: 0.8f32,
            width1x_: 0,
            sheight1x_: 0,
            width_scaled_: 0,
            height_scaled_: 0,
            buf_img_width_: WIDTH,
            buf_img_height_: HEIGHT,
            buf_scaled_width_: WIDTH,
            buf_scaled_height_: HEIGHT,
            buf_img: vec![0; WIDTH * HEIGHT],
            buf_img_scaled: vec![0; WIDTH * HEIGHT],
        }
    }

    fn set_scale_step(&mut self, step: f32) {
        if step > 0.0 && step <= 1.0 {
            self.scale_step = step;
        }
    }

    fn set_min_scale(&mut self, min_scale: f32) {
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
}


// src/util/image_pyramid.cpp

const seeta::ImageData* ImagePyramid::GetNextScaleImage(float* scale_factor) {
  if (scale_factor_ >= min_scale_) {
    if (scale_factor != nullptr)
      *scale_factor = scale_factor_;

    width_scaled_ = static_cast<int32_t>(width1x_ * scale_factor_);
    height_scaled_ = static_cast<int32_t>(height1x_ * scale_factor_);

    seeta::ImageData src_img(width1x_, height1x_);
    seeta::ImageData dest_img(width_scaled_, height_scaled_);
    src_img.data = buf_img_;
    dest_img.data = buf_img_scaled_;
    seeta::fd::ResizeImage(src_img, &dest_img);
    scale_factor_ *= scale_step_;

    img_scaled_.data = buf_img_scaled_;
    img_scaled_.width = width_scaled_;
    img_scaled_.height = height_scaled_;
    return &img_scaled_;
  } else {
    return nullptr;
  }
}

void ImagePyramid::SetImage1x(const uint8_t* img_data, int32_t width,
    int32_t height) {
  if (width > buf_img_width_ || height > buf_img_height_) {
    delete[] buf_img_;

    buf_img_width_ = width;
    buf_img_height_ = height;
    buf_img_ = new uint8_t[width * height];
  }

  width1x_ = width;
  height1x_ = height;
  std::memcpy(buf_img_, img_data, width * height * sizeof(uint8_t));
  scale_factor_ = max_scale_;
  UpdateBufScaled();
}

void ImagePyramid::UpdateBufScaled() {
  if (width1x_ == 0 || height1x_ == 0)
    return;

  int32_t max_width = static_cast<int32_t>(width1x_ * max_scale_ + 0.5);
  int32_t max_height = static_cast<int32_t>(height1x_ * max_scale_ + 0.5);

  if (max_width > buf_scaled_width_ || max_height > buf_scaled_height_) {
    delete[] buf_img_scaled_;

    buf_scaled_width_ = max_width;
    buf_scaled_height_ = max_height;
    buf_img_scaled_ = new uint8_t[max_width * max_height];

    img_scaled_.data = nullptr;
    img_scaled_.width = 0;
    img_scaled_.height = 0;
  }
}


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


// src/feat/lab_feature_map.cpp

void LABFeatureMap::Compute(const uint8_t* input, int32_t width,
    int32_t height) {
  if (input == nullptr || width <= 0 || height <= 0) {
    return;  // @todo handle the errors!!!
  }

  Reshape(width, height);
  ComputeIntegralImages(input);
  ComputeRectSum();
  ComputeFeatureMap();
}

float LABFeatureMap::GetStdDev() const {
  double mean;
  double m2;
  double area = roi_.width * roi_.height;

  int32_t top_left;
  int32_t top_right;
  int32_t bottom_left;
  int32_t bottom_right;

  if (roi_.x != 0) {
    if (roi_.y != 0) {
      top_left = (roi_.y - 1) * width_ + roi_.x - 1;
      top_right = top_left + roi_.width;
      bottom_left = top_left + roi_.height * width_;
      bottom_right = bottom_left + roi_.width;

      mean = (int_img_[bottom_right] - int_img_[bottom_left] +
        int_img_[top_left] - int_img_[top_right]) / area;
      m2 = (square_int_img_[bottom_right] - square_int_img_[bottom_left] +
        square_int_img_[top_left] - square_int_img_[top_right]) / area;
    } else {
      bottom_left = (roi_.height - 1) * width_ + roi_.x - 1;
      bottom_right = bottom_left + roi_.width;

      mean = (int_img_[bottom_right] - int_img_[bottom_left]) / area;
      m2 = (square_int_img_[bottom_right] - square_int_img_[bottom_left]) / area;
    }
  } else {
    if (roi_.y != 0) {
      top_right = (roi_.y - 1) * width_ + roi_.width - 1;
      bottom_right = top_right + roi_.height * width_;

      mean = (int_img_[bottom_right] - int_img_[top_right]) / area;
      m2 = (square_int_img_[bottom_right] - square_int_img_[top_right]) / area;
    } else {
      bottom_right = (roi_.height - 1) * width_ + roi_.width - 1;
      mean = int_img_[bottom_right] / area;
      m2 = square_int_img_[bottom_right] / area;
    }
  }

  return static_cast<float>(std::sqrt(m2 - mean * mean));
}

void LABFeatureMap::Reshape(int32_t width, int32_t height) {
  width_ = width;
  height_ = height;

  int32_t len = width_ * height_;
  feat_map_.resize(len);
  rect_sum_.resize(len);
  int_img_.resize(len);
  square_int_img_.resize(len);
}

void LABFeatureMap::ComputeIntegralImages(const uint8_t* input) {
  int32_t len = width_ * height_;

  seeta::fd::MathFunction::UInt8ToInt32(input, int_img_.data(), len);
  seeta::fd::MathFunction::Square(int_img_.data(), square_int_img_.data(), len);
  Integral(int_img_.data());
  Integral(square_int_img_.data());
}

void LABFeatureMap::ComputeRectSum() {
  int32_t width = width_ - rect_width_;
  int32_t height = height_ - rect_height_;
  const int32_t* int_img = int_img_.data();
  int32_t* rect_sum = rect_sum_.data();

  *rect_sum = *(int_img + (rect_height_ - 1) * width_ + rect_width_ - 1);
  seeta::fd::MathFunction::VectorSub(int_img + (rect_height_ - 1) * width_ +
    rect_width_, int_img + (rect_height_ - 1) * width_, rect_sum + 1, width);

#pragma omp parallel num_threads(SEETA_NUM_THREADS)
  {
#pragma omp for nowait
    for (int32_t i = 1; i <= height; i++) {
      const int32_t* top_left = int_img + (i - 1) * width_;
      const int32_t* top_right = top_left + rect_width_ - 1;
      const int32_t* bottom_left = top_left + rect_height_ * width_;
      const int32_t* bottom_right = bottom_left + rect_width_ - 1;
      int32_t* dest = rect_sum + i * width_;

      *(dest++) = (*bottom_right) - (*top_right);
      seeta::fd::MathFunction::VectorSub(bottom_right + 1, top_right + 1, dest, width);
      seeta::fd::MathFunction::VectorSub(dest, bottom_left, dest, width);
      seeta::fd::MathFunction::VectorAdd(dest, top_left, dest, width);
    }
  }
}

void LABFeatureMap::ComputeFeatureMap() {
  int32_t width = width_ - rect_width_ * num_rect_;
  int32_t height = height_ - rect_height_ * num_rect_;
  int32_t offset = width_ * rect_height_;
  uint8_t* feat_map = feat_map_.data();

#pragma omp parallel num_threads(SEETA_NUM_THREADS)
  {
#pragma omp for nowait
    for (int32_t r = 0; r <= height; r++) {
      for (int32_t c = 0; c <= width; c++) {
        uint8_t* dest = feat_map + r * width_ + c;
        *dest = 0;

        int32_t white_rect_sum = rect_sum_[(r + rect_height_) * width_ + c + rect_width_];
        int32_t black_rect_idx = r * width_ + c;
        *dest |= (white_rect_sum >= rect_sum_[black_rect_idx] ? 0x80 : 0x0);
        black_rect_idx += rect_width_;
        *dest |= (white_rect_sum >= rect_sum_[black_rect_idx] ? 0x40 : 0x0);
        black_rect_idx += rect_width_;
        *dest |= (white_rect_sum >= rect_sum_[black_rect_idx] ? 0x20 : 0x0);
        black_rect_idx += offset;
        *dest |= (white_rect_sum >= rect_sum_[black_rect_idx] ? 0x08 : 0x0);
        black_rect_idx += offset;
        *dest |= (white_rect_sum >= rect_sum_[black_rect_idx] ? 0x01 : 0x0);
        black_rect_idx -= rect_width_;
        *dest |= (white_rect_sum >= rect_sum_[black_rect_idx] ? 0x02 : 0x0);
        black_rect_idx -= rect_width_;
        *dest |= (white_rect_sum >= rect_sum_[black_rect_idx] ? 0x04 : 0x0);
        black_rect_idx -= offset;
        *dest |= (white_rect_sum >= rect_sum_[black_rect_idx] ? 0x10 : 0x0);
      }
    }
  }
}


// src/feat/surf_feature_map.cpp

void SURFFeaturePool::Create() {
  if (sample_height_ - patch_min_height_ <= sample_width_ - patch_min_width_) {
    for (size_t i = 0; i < format_.size(); i++) {
      const SURFPatchFormat & format = format_[i];
      for (int32_t h = patch_min_height_; h <= sample_height_;
          h += patch_size_inc_step_) {
        if (h % format.num_cell_per_col != 0 || h % format.height != 0)
          continue;
        int32_t w = h / format.height * format.width;
        if (w % format.num_cell_per_row != 0 || w < patch_min_width_ ||
            w > sample_width_)
          continue;
        AddAllFeaturesToPool(w, h, format.num_cell_per_row,
          format.num_cell_per_col);
      }
    }
  } else {
    for (size_t i = 0; i < format_.size(); i++) {
      const SURFPatchFormat & format = format_[i];
      for (int32_t w = patch_min_width_; w <= patch_min_width_;
          w += patch_size_inc_step_) {
        if (w % format.num_cell_per_row != 0 || w % format.width != 0)
          continue;
        int32_t h = w / format.width * format.height;
        if (h % format.num_cell_per_col != 0 || h < patch_min_height_ ||
            h > sample_height_)
          continue;
        AddAllFeaturesToPool(w, h, format.num_cell_per_row,
          format.num_cell_per_col);
      }
    }
  }
}

void SURFFeaturePool::AddPatchFormat(int32_t width, int32_t height,
    int32_t num_cell_per_row, int32_t num_cell_per_col) {
  for (size_t i = 0; i < format_.size(); i++) {
    const SURFPatchFormat & format = format_[i];
    if (format.height == height &&
      format.width == width &&
      format.num_cell_per_row == num_cell_per_row &&
      format.num_cell_per_col == num_cell_per_col)
      return;
  }

  SURFPatchFormat new_format;
  new_format.height = height;
  new_format.width = width;
  new_format.num_cell_per_row = num_cell_per_row;
  new_format.num_cell_per_col = num_cell_per_col;
  format_.push_back(new_format);
}

void SURFFeaturePool::AddAllFeaturesToPool(int32_t width, int32_t height,
    int32_t num_cell_per_row, int32_t num_cell_per_col) {
  SURFFeature feat;
  feat.patch.width = width;
  feat.patch.height = height;
  feat.num_cell_per_row = num_cell_per_row;
  feat.num_cell_per_col = num_cell_per_col;

  for (int32_t y = 0; y <= sample_height_ - height; y += patch_move_step_y_) {
    feat.patch.y = y;
    for (int32_t x = 0; x <= sample_width_ - width; x += patch_move_step_x_) {
      feat.patch.x = x;
      pool_.push_back(feat);
    }
  }
}

void SURFFeatureMap::Compute(const uint8_t* input, int32_t width,
    int32_t height) {
  if (input == nullptr || width <= 0 || height <= 0) {
    return;  // @todo handle the error!
  }
  Reshape(width, height);
  ComputeGradientImages(input);
  ComputeIntegralImages();
}

void SURFFeatureMap::GetFeatureVector(int32_t feat_id, float* feat_vec) {
  if (buf_valid_[feat_id] == 0) {
    ComputeFeatureVector(feat_pool_[feat_id], feat_vec_buf_[feat_id].data());
    NormalizeFeatureVectorL2(feat_vec_buf_[feat_id].data(),
      feat_vec_normed_buf_[feat_id].data(),
      static_cast<int32_t>(feat_vec_normed_buf_[feat_id].size()));
    buf_valid_[feat_id] = 1;
    buf_valid_reset_ = true;
  }

  std::memcpy(feat_vec, feat_vec_normed_buf_[feat_id].data(),
    feat_vec_normed_buf_[feat_id].size() * sizeof(float));
}

void SURFFeatureMap::InitFeaturePool() {
  feat_pool_.AddPatchFormat(1, 1, 2, 2);
  feat_pool_.AddPatchFormat(1, 2, 2, 2);
  feat_pool_.AddPatchFormat(2, 1, 2, 2);
  feat_pool_.AddPatchFormat(2, 3, 2, 2);
  feat_pool_.AddPatchFormat(3, 2, 2, 2);
  feat_pool_.Create();

  int32_t feat_pool_size = static_cast<int32_t>(feat_pool_.size());
  feat_vec_buf_.resize(feat_pool_size);
  feat_vec_normed_buf_.resize(feat_pool_size);
  for (size_t i = 0; i < feat_pool_size; i++) {
    int32_t dim = GetFeatureVectorDim(static_cast<int32_t>(i));
    feat_vec_buf_[i].resize(dim);
    feat_vec_normed_buf_[i].resize(dim);
  }
  buf_valid_.resize(feat_pool_size, 0);
}

void SURFFeatureMap::Reshape(int32_t width, int32_t height) {
  width_ = width;
  height_ = height;

  int32_t len = width_ * height_;
  grad_x_.resize(len);
  grad_y_.resize(len);
  int_img_.resize(len * kNumIntChannel);
  img_buf_.resize(len);
}

void SURFFeatureMap::ComputeGradientImages(const uint8_t* input) {
  int32_t len = width_ * height_;
  seeta::fd::MathFunction::UInt8ToInt32(input, img_buf_.data(), len);
  ComputeGradX(img_buf_.data());
  ComputeGradY(img_buf_.data());
}

void SURFFeatureMap::ComputeGradX(const int32_t* input) {
  int32_t* dx = grad_x_.data();
  int32_t len = width_ - 2;

#pragma omp parallel num_threads(SEETA_NUM_THREADS)
  {
#pragma omp for nowait
    for (int32_t r = 0; r < height_; r++) {
      const int32_t* src = input + r * width_;
      int32_t* dest = dx + r * width_;
      *dest = ((*(src + 1)) - (*src)) << 1;
      seeta::fd::MathFunction::VectorSub(src + 2, src, dest + 1, len);
      dest += (width_ - 1);
      src += (width_ - 1);
      *dest = ((*src) - (*(src - 1))) << 1;
    }
  }
}

void SURFFeatureMap::ComputeGradY(const int32_t* input) {
  int32_t* dy = grad_y_.data();
  int32_t len = width_;
  seeta::fd::MathFunction::VectorSub(input + width_, input, dy, len);
  seeta::fd::MathFunction::VectorAdd(dy, dy, dy, len);

#pragma omp parallel num_threads(SEETA_NUM_THREADS)
  {
#pragma omp for nowait
    for (int32_t r = 1; r < height_ - 1; r++) {
      const int32_t* src = input + (r - 1) * width_;
      int32_t* dest = dy + r * width_;
      seeta::fd::MathFunction::VectorSub(src + (width_ << 1), src, dest, len);
    }
  }
  int32_t offset = (height_ - 1) * width_;
  dy += offset;
  seeta::fd::MathFunction::VectorSub(input + offset, input + offset - width_,
    dy, len);
  seeta::fd::MathFunction::VectorAdd(dy, dy, dy, len);
}

void SURFFeatureMap::ComputeIntegralImages() {
  FillIntegralChannel(grad_x_.data(), 0);
  FillIntegralChannel(grad_y_.data(), 4);

  int32_t len = width_ * height_;
  seeta::fd::MathFunction::VectorAbs(grad_x_.data(), img_buf_.data(), len);
  FillIntegralChannel(img_buf_.data(), 1);
  seeta::fd::MathFunction::VectorAbs(grad_y_.data(), img_buf_.data(), len);
  FillIntegralChannel(img_buf_.data(), 5);
  MaskIntegralChannel();
  Integral();
}

void SURFFeatureMap::MaskIntegralChannel() {
  const int32_t* grad_x = grad_x_.data();
  const int32_t* grad_y = grad_y_.data();
  int32_t len = width_ * height_;
#ifdef USE_SSE
  __m128i dx;
  __m128i dy;
  __m128i dx_mask;
  __m128i dy_mask;
  __m128i zero = _mm_set1_epi32(0);
  __m128i xor_bits = _mm_set_epi32(0x0, 0x0, 0xffffffff, 0xffffffff);
  __m128i data;
  __m128i result;
  __m128i* src = reinterpret_cast<__m128i*>(int_img_.data());

  for (int32_t i = 0; i < len; i++) {
    dx = _mm_set1_epi32(*(grad_x++));
    dy = _mm_set1_epi32(*(grad_y++));
    dx_mask = _mm_xor_si128(_mm_cmplt_epi32(dx, zero), xor_bits);
    dy_mask = _mm_xor_si128(_mm_cmplt_epi32(dy, zero), xor_bits);

    data = _mm_loadu_si128(src);
    result = _mm_and_si128(data, dy_mask);
    _mm_storeu_si128(src++, result);
    data = _mm_loadu_si128(src);
    result = _mm_and_si128(data, dx_mask);
    _mm_storeu_si128(src++, result);
  }
#else
  int32_t dx, dy, dx_mask, dy_mask, cmp;
  int32_t xor_bits[] = {-1, -1, 0, 0};

  int32_t* src = int_img_.data();
  for (int32_t i = 0; i < len; i++) {
      dy = *(grad_y++);
      dx = *(grad_x++);

      cmp = dy < 0 ? 0xffffffff : 0x0;
      for (int32_t j = 0; j < 4; j++) {
          // cmp xor xor_bits
          dy_mask = cmp ^ xor_bits[j];
          *(src) = (*src) & dy_mask;
          src++;
      }

      cmp = dx < 0 ? 0xffffffff : 0x0;
      for (int32_t j = 0; j < 4; j++) {
          // cmp xor xor_bits
          dx_mask = cmp ^ xor_bits[j];
          *(src) = (*src) & dx_mask;
          src++;
      }
  }
#endif
}

void SURFFeatureMap::Integral() {
  int32_t* data = int_img_.data();
  int32_t len = kNumIntChannel * width_;

  // Cummulative sum by row
  for (int32_t r = 0; r < height_ - 1; r++) {
    int32_t* row1 = data + r * len;
    int32_t* row2 = row1 + len;
    seeta::fd::MathFunction::VectorAdd(row1, row2, row2, len);
  }
  // Cummulative sum by column
  for (int32_t r = 0; r < height_; r++)
    VectorCumAdd(data + r * len, len, kNumIntChannel);
}

void SURFFeatureMap::VectorCumAdd(int32_t* x, int32_t len,
    int32_t num_channel) {
#ifdef USE_SSE
  __m128i x1;
  __m128i y1;
  __m128i z1;
  __m128i* x2 = reinterpret_cast<__m128i*>(x);
  __m128i* y2 = reinterpret_cast<__m128i*>(x + num_channel);
  __m128i* z2 = y2;

  len = len / num_channel - 1;
  for (int32_t i = 0; i < len; i++) {
    // first 4 channels
    x1 = _mm_loadu_si128(x2++);
    y1 = _mm_loadu_si128(y2++);
    z1 = _mm_add_epi32(x1, y1);
    _mm_storeu_si128(z2, z1);
    z2 = y2;

    // second 4 channels
    x1 = _mm_loadu_si128(x2++);
    y1 = _mm_loadu_si128(y2++);
    z1 = _mm_add_epi32(x1, y1);
    _mm_storeu_si128(z2, z1);
    z2 = y2;
  }
#else
  int32_t cols = len / num_channel - 1;
  for (int32_t i = 0; i < cols; i++) {
    int32_t* col1 = x + i * num_channel;
    int32_t* col2 = col1 + num_channel;
    seeta::fd::MathFunction::VectorAdd(col1, col2, col2, num_channel);
  }
#endif
}

void SURFFeatureMap::ComputeFeatureVector(const SURFFeature & feat,
    int32_t* feat_vec) {
  int32_t init_cell_x = roi_.x + feat.patch.x;
  int32_t init_cell_y = roi_.y + feat.patch.y;
  int32_t cell_width = feat.patch.width / feat.num_cell_per_row * kNumIntChannel;
  int32_t cell_height = feat.patch.height / feat.num_cell_per_col;
  int32_t row_width = width_ * kNumIntChannel;
  const int32_t* cell_top_left[kNumIntChannel];
  const int32_t* cell_top_right[kNumIntChannel];
  const int32_t* cell_bottom_left[kNumIntChannel];
  const int32_t* cell_bottom_right[kNumIntChannel];
  int* feat_val = feat_vec;
  const int32_t* int_img = int_img_.data();
  int32_t offset = 0;

  if (init_cell_y != 0) {
    if (init_cell_x != 0) {
      const int32_t* tmp_cell_top_right[kNumIntChannel];

      // cell #1
      offset = row_width * (init_cell_y - 1) +
        (init_cell_x - 1) * kNumIntChannel;
      for (int32_t i = 0; i < kNumIntChannel; i++) {
        cell_top_left[i] = int_img + (offset++);
        cell_top_right[i] = cell_top_left[i] + cell_width;
        cell_bottom_left[i] = cell_top_left[i] + row_width * cell_height;
        cell_bottom_right[i] = cell_bottom_left[i] + cell_width;
        *(feat_val++) = *(cell_bottom_right[i]) + *(cell_top_left[i]) -
                        *(cell_top_right[i]) - *(cell_bottom_left[i]);
        tmp_cell_top_right[i] = cell_bottom_right[i];
      }

      // cells in 1st row
      for (int32_t i = 1; i < feat.num_cell_per_row; i++) {
        for (int32_t j = 0; j < kNumIntChannel; j++) {
          cell_top_left[j] = cell_top_right[j];
          cell_top_right[j] += cell_width;
          cell_bottom_left[j] = cell_bottom_right[j];
          cell_bottom_right[j] += cell_width;
          *(feat_val++) = *(cell_bottom_right[j]) + *(cell_top_left[j]) -
                          *(cell_top_right[j]) - *(cell_bottom_left[j]);
        }
      }

      for (int32_t i = 0; i < kNumIntChannel; i++)
        cell_top_right[i] = tmp_cell_top_right[i];
    } else {
      const int32_t* tmp_cell_top_right[kNumIntChannel];

      // cell #1
      offset = row_width * (init_cell_y - 1) + cell_width - kNumIntChannel;
      for (int32_t i = 0; i < kNumIntChannel; i++) {
        cell_top_right[i] = int_img + (offset++);
        cell_bottom_right[i] = cell_top_right[i] + row_width * cell_height;
        tmp_cell_top_right[i] = cell_bottom_right[i];
        *(feat_val++) = *(cell_bottom_right[i]) - *(cell_top_right[i]);
      }

      // cells in 1st row
      for (int32_t i = 1; i < feat.num_cell_per_row; i++) {
        for (int32_t j = 0; j < kNumIntChannel; j++) {
          cell_top_left[j] = cell_top_right[j];
          cell_top_right[j] += cell_width;
          cell_bottom_left[j] = cell_bottom_right[j];
          cell_bottom_right[j] += cell_width;
          *(feat_val++) = *(cell_bottom_right[j]) + *(cell_top_left[j]) -
                          *(cell_top_right[j]) - *(cell_bottom_left[j]);
        }
      }

      for (int32_t i = 0; i < kNumIntChannel; i++)
        cell_top_right[i] = tmp_cell_top_right[i];
    }
  } else {
    if (init_cell_x != 0) {
      // cell #1
      offset = row_width * (cell_height - 1) +
        (init_cell_x - 1) * kNumIntChannel;
      for (int32_t i = 0; i < kNumIntChannel; i++) {
        cell_bottom_left[i] = int_img + (offset++);
        cell_bottom_right[i] = cell_bottom_left[i] + cell_width;
        *(feat_val++) = *(cell_bottom_right[i]) - *(cell_bottom_left[i]);
        cell_top_right[i] = cell_bottom_right[i];
      }

      // cells in 1st row
      for (int32_t i = 1; i < feat.num_cell_per_row; i++) {
        for (int32_t j = 0; j < kNumIntChannel; j++) {
          cell_bottom_left[j] = cell_bottom_right[j];
          cell_bottom_right[j] += cell_width;
          *(feat_val++) = *(cell_bottom_right[j]) - *(cell_bottom_left[j]);
        }
      }
    } else {
      // cell #1
      offset = row_width * (cell_height - 1) + cell_width - kNumIntChannel;
      for (int32_t i = 0; i < kNumIntChannel; i++) {
        cell_bottom_right[i] = int_img + (offset++);
        *(feat_val++) = *(cell_bottom_right[i]);
        cell_top_right[i] = cell_bottom_right[i];
      }

      // cells in 1st row
      for (int32_t i = 1; i < feat.num_cell_per_row; i++) {
        for (int32_t j = 0; j < kNumIntChannel; j++) {
          cell_bottom_left[j] = cell_bottom_right[j];
          cell_bottom_right[j] += cell_width;
          *(feat_val++) = *(cell_bottom_right[j]) - *(cell_bottom_left[j]);
        }
      }
    }
  }

  // from BR of last cell in current row to BR of first cell in next row
  offset = cell_height * row_width - feat.patch.width *
    kNumIntChannel + cell_width;

  // cells in following rows
  for (int32_t i = 1; i < feat.num_cell_per_row; i++) {
    // cells in 1st column
    if (init_cell_x == 0) {
      for (int32_t j = 0; j < kNumIntChannel; j++) {
        cell_bottom_right[j] += offset;
        *(feat_val++) = *(cell_bottom_right[j]) - *(cell_top_right[j]);
      }
    } else {
      for (int32_t j = 0; j < kNumIntChannel; j++) {
        cell_bottom_right[j] += offset;
        cell_top_left[j] = cell_top_right[j] - cell_width;
        cell_bottom_left[j] = cell_bottom_right[j] - cell_width;
        *(feat_val++) = *(cell_bottom_right[j]) + *(cell_top_left[j]) -
                        *(cell_top_right[j]) - *(cell_bottom_left[j]);
      }
    }

    // cells in following columns
    for (int32_t j = 1; j < feat.num_cell_per_row; j++) {
      for (int32_t k = 0; k < kNumIntChannel; k++) {
        cell_top_left[k] = cell_top_right[k];
        cell_top_right[k] += cell_width;

        cell_bottom_left[k] = cell_bottom_right[k];
        cell_bottom_right[k] += cell_width;

        *(feat_val++) = *(cell_bottom_right[k]) + *(cell_top_left[k]) -
                        *(cell_bottom_left[k]) - *(cell_top_right[k]);
      }
    }

    for (int32_t j = 0; j < kNumIntChannel; j++)
      cell_top_right[j] += offset;
  }
}

void SURFFeatureMap::NormalizeFeatureVectorL2(const int32_t* feat_vec,
    float* feat_vec_normed, int32_t len) const {
  double prod = 0.0;
  float norm_l2 = 0.0f;

  for (int32_t i = 0; i < len; i++)
    prod += static_cast<double>(feat_vec[i] * feat_vec[i]);
  if (prod != 0) {
    norm_l2 = static_cast<float>(std::sqrt(prod));
    for (int32_t i = 0; i < len; i++)
      feat_vec_normed[i] = feat_vec[i] / norm_l2;
  } else {
    for (int32_t i = 0; i < len; i++)
      feat_vec_normed[i] = 0.0f;
  }
}


// src/classifier/lab_boosted_classifier.cpp

void LABBaseClassifier::SetWeights(const float* weights, int32_t num_bin) {
  weights_.resize(num_bin + 1);
  num_bin_ = num_bin;
  std::copy(weights, weights + num_bin_ + 1, weights_.begin());
}

bool LABBoostedClassifier::Classify(float* score, float* outputs) {
  bool isPos = true;
  float s = 0.0f;

  for (size_t i = 0; isPos && i < base_classifiers_.size();) {
    for (int32_t j = 0; j < kFeatGroupSize; j++, i++) {
      uint8_t featVal = feat_map_->GetFeatureVal(feat_[i].x, feat_[i].y);
      s += base_classifiers_[i]->weights(featVal);
    }
    if (s < base_classifiers_[i - 1]->threshold())
      isPos = false;
  }
  isPos = isPos && ((!use_std_dev_) || feat_map_->GetStdDev() > kStdDevThresh);

  if (score != nullptr)
    *score = s;
  if (outputs != nullptr)
    *outputs = s;

  return isPos;
}

void LABBoostedClassifier::AddFeature(int32_t x, int32_t y) {
  LABFeature feat;
  feat.x = x;
  feat.y = y;
  feat_.push_back(feat);
}

void LABBoostedClassifier::AddBaseClassifier(const float* weights,
    int32_t num_bin, float thresh) {
  std::shared_ptr<LABBaseClassifier> classifier(new LABBaseClassifier());
  classifier->SetWeights(weights, num_bin);
  classifier->SetThreshold(thresh);
  base_classifiers_.push_back(classifier);
}


// src/classifier/surf_mlp.cpp

bool SURFMLP::Classify(float* score, float* outputs) {
  float* dest = input_buf_.data();
  for (size_t i = 0; i < feat_id_.size(); i++) {
    feat_map_->GetFeatureVector(feat_id_[i] - 1, dest);
    dest += feat_map_->GetFeatureVectorDim(feat_id_[i]);
  }
  output_buf_.resize(model_->GetOutputDim());
  model_->Compute(input_buf_.data(), output_buf_.data());

  if (score != nullptr)
    *score = output_buf_[0];
  if (outputs != nullptr) {
    std::memcpy(outputs, output_buf_.data(),
      model_->GetOutputDim() * sizeof(float));
  }

  return (output_buf_[0] > thresh_);
}

void SURFMLP::AddFeatureByID(int32_t feat_id) {
  feat_id_.push_back(feat_id);
}

void SURFMLP::AddLayer(int32_t input_dim, int32_t output_dim,
    const float* weights, const float* bias, bool is_output) {
  if (model_->GetLayerNum() == 0)
    input_buf_.resize(input_dim);
  model_->AddLayer(input_dim, output_dim, weights, bias, is_output);
}


// src/classifier/mlp.cpp

void MLPLayer::Compute(const float* input, float* output) {
#pragma omp parallel num_threads(SEETA_NUM_THREADS)
  {
#pragma omp for nowait
    for (int32_t i = 0; i < output_dim_; i++) {
      output[i] = seeta::fd::MathFunction::VectorInnerProduct(input,
        weights_.data() + i * input_dim_, input_dim_) + bias_[i];
      output[i] = (act_func_type_ == 1 ? ReLU(output[i]) : Sigmoid(-output[i]));
    }
  }
}

void MLP::Compute(const float* input, float* output) {
  layer_buf_[0].resize(layers_[0]->GetOutputDim());
  layers_[0]->Compute(input, layer_buf_[0].data());

  size_t i; /**< layer index */
  for (i = 1; i < layers_.size() - 1; i++) {
    layer_buf_[i % 2].resize(layers_[i]->GetOutputDim());
    layers_[i]->Compute(layer_buf_[(i + 1) % 2].data(), layer_buf_[i % 2].data());
  }
  layers_.back()->Compute(layer_buf_[(i + 1) % 2].data(), output);
}

void MLP::AddLayer(int32_t inputDim, int32_t outputDim, const float* weights,
    const float* bias, bool is_output) {
  if (layers_.size() > 0 && inputDim != layers_.back()->GetOutputDim())
    return;  // @todo handle the errors!!!

  std::shared_ptr<seeta::fd::MLPLayer> layer(new seeta::fd::MLPLayer(is_output ? 0 : 1));
  layer->SetSize(inputDim, outputDim);
  layer->SetWeights(weights, inputDim * outputDim);
  layer->SetBias(bias, outputDim);
  layers_.push_back(layer);
}


// src/io/lab_boost_model_reader.cpp

bool LABBoostModelReader::Read(std::istream* input,
    seeta::fd::Classifier* model) {
  bool is_read;
  seeta::fd::LABBoostedClassifier* lab_boosted_classifier =
    dynamic_cast<seeta::fd::LABBoostedClassifier*>(model);

  input->read(reinterpret_cast<char*>(&num_base_classifer_), sizeof(int32_t));
  input->read(reinterpret_cast<char*>(&num_bin_), sizeof(int32_t));

  is_read = (!input->fail()) && num_base_classifer_ > 0 && num_bin_ > 0 &&
    ReadFeatureParam(input, lab_boosted_classifier) &&
    ReadBaseClassifierParam(input, lab_boosted_classifier);

  return is_read;
}

bool LABBoostModelReader::ReadFeatureParam(std::istream* input,
    seeta::fd::LABBoostedClassifier* model) {
  int32_t x;
  int32_t y;
  for (int32_t i = 0; i < num_base_classifer_; i++) {
    input->read(reinterpret_cast<char*>(&x), sizeof(int32_t));
    input->read(reinterpret_cast<char*>(&y), sizeof(int32_t));
    model->AddFeature(x, y);
  }

  return !input->fail();
}

bool LABBoostModelReader::ReadBaseClassifierParam(std::istream* input,
    seeta::fd::LABBoostedClassifier* model) {
  std::vector<float> thresh;
  thresh.resize(num_base_classifer_);
  input->read(reinterpret_cast<char*>(thresh.data()),
    sizeof(float)* num_base_classifer_);

  int32_t weight_len = sizeof(float)* (num_bin_ + 1);
  std::vector<float> weights;
  weights.resize(num_bin_ + 1);
  for (int32_t i = 0; i < num_base_classifer_; i++) {
    input->read(reinterpret_cast<char*>(weights.data()), weight_len);
    model->AddBaseClassifier(weights.data(), num_bin_, thresh[i]);
  }

  return !input->fail();
}


// src/io/surf_mlp_model_reader.cpp

bool SURFMLPModelReader::Read(std::istream* input,
    seeta::fd::Classifier* model) {
  bool is_read = false;
  seeta::fd::SURFMLP* surf_mlp = dynamic_cast<seeta::fd::SURFMLP*>(model);
  int32_t num_layer;
  int32_t num_feat;
  int32_t input_dim;
  int32_t output_dim;
  float thresh;

  input->read(reinterpret_cast<char*>(&num_layer), sizeof(int32_t));
  if (num_layer <= 0) {
    is_read = false;  // @todo handle the errors and the following ones!!!
  }
  input->read(reinterpret_cast<char*>(&num_feat), sizeof(int32_t));
  if (num_feat <= 0) {
    is_read = false;
  }

  feat_id_buf_.resize(num_feat);
  input->read(reinterpret_cast<char*>(feat_id_buf_.data()),
    sizeof(int32_t) * num_feat);
  for (int32_t i = 0; i < num_feat; i++)
    surf_mlp->AddFeatureByID(feat_id_buf_[i]);

  input->read(reinterpret_cast<char*>(&thresh), sizeof(float));
  surf_mlp->SetThreshold(thresh);
  input->read(reinterpret_cast<char*>(&input_dim), sizeof(int32_t));
  if (input_dim <= 0) {
    is_read = false;
  }

  for (int32_t i = 1; i < num_layer; i++) {
    input->read(reinterpret_cast<char*>(&output_dim), sizeof(int32_t));
    if (output_dim <= 0) {
      is_read = false;
    }

    int32_t len = input_dim * output_dim;
    weights_buf_.resize(len);
    input->read(reinterpret_cast<char*>(weights_buf_.data()),
      sizeof(float) * len);

    bias_buf_.resize(output_dim);
    input->read(reinterpret_cast<char*>(bias_buf_.data()),
      sizeof(float) * output_dim);

    if (i < num_layer - 1) {
      surf_mlp->AddLayer(input_dim, output_dim, weights_buf_.data(),
        bias_buf_.data());
    } else {
      surf_mlp->AddLayer(input_dim, output_dim, weights_buf_.data(),
        bias_buf_.data(), true);
    }
    input_dim = output_dim;
  }

  is_read = !input->fail();

  return is_read;
}


// src/fust.cpp

bool FuStDetector::LoadModel(const std::string & model_path) {
  std::ifstream model_file(model_path, std::ifstream::binary);
  bool is_loaded = true;

  if (!model_file.is_open()) {
    is_loaded = false;
  } else {
    hierarchy_size_.clear();
    num_stage_.clear();
    wnd_src_id_.clear();

    int32_t hierarchy_size;
    int32_t num_stage;
    int32_t num_wnd_src;
    int32_t type_id;
    int32_t feat_map_index = 0;
    std::shared_ptr<seeta::fd::ModelReader> reader;
    std::shared_ptr<seeta::fd::Classifier> classifier;
    seeta::fd::ClassifierType classifier_type;

    model_file.read(reinterpret_cast<char*>(&num_hierarchy_), sizeof(int32_t));
    for (int32_t i = 0; is_loaded && i < num_hierarchy_; i++) {
      model_file.read(reinterpret_cast<char*>(&hierarchy_size),
        sizeof(int32_t));
      hierarchy_size_.push_back(hierarchy_size);

      for (int32_t j = 0; is_loaded && j < hierarchy_size; j++) {
        model_file.read(reinterpret_cast<char*>(&num_stage), sizeof(int32_t));
        num_stage_.push_back(num_stage);

        for (int32_t k = 0; is_loaded && k < num_stage; k++) {
          model_file.read(reinterpret_cast<char*>(&type_id), sizeof(int32_t));
          classifier_type = static_cast<seeta::fd::ClassifierType>(type_id);
          reader = CreateModelReader(classifier_type);
          classifier = CreateClassifier(classifier_type);

          is_loaded = !model_file.fail() &&
            reader->Read(&model_file, classifier.get());
          if (is_loaded) {
            model_.push_back(classifier);
            std::shared_ptr<seeta::fd::FeatureMap> feat_map;
            if (cls2feat_idx_.count(classifier_type) == 0) {
              feat_map_.push_back(CreateFeatureMap(classifier_type));
              cls2feat_idx_.insert(
                std::map<seeta::fd::ClassifierType, int32_t>::value_type(
                classifier_type, feat_map_index++));
            }
            feat_map = feat_map_[cls2feat_idx_.at(classifier_type)];
            model_.back()->SetFeatureMap(feat_map.get());
          }
        }

        wnd_src_id_.push_back(std::vector<int32_t>());
        model_file.read(reinterpret_cast<char*>(&num_wnd_src), sizeof(int32_t));
        if (num_wnd_src > 0) {
          wnd_src_id_.back().resize(num_wnd_src);
          for (int32_t k = 0; k < num_wnd_src; k++) {
            model_file.read(reinterpret_cast<char*>(&(wnd_src_id_.back()[k])),
              sizeof(int32_t));
          }
        }
      }
    }

    model_file.close();
  }

  return is_loaded;
}

std::vector<seeta::FaceInfo> FuStDetector::Detect(
    seeta::fd::ImagePyramid* img_pyramid) {
  float score;
  seeta::FaceInfo wnd_info;
  seeta::Rect wnd;
  float scale_factor = 0.0;
  const seeta::ImageData* img_scaled =
    img_pyramid->GetNextScaleImage(&scale_factor);

  wnd.height = wnd.width = wnd_size_;

  // Sliding window

  std::vector<std::vector<seeta::FaceInfo> > proposals(hierarchy_size_[0]);
  std::shared_ptr<seeta::fd::FeatureMap> & feat_map_1 =
    feat_map_[cls2feat_idx_[model_[0]->type()]];

  while (img_scaled != nullptr) {
    feat_map_1->Compute(img_scaled->data, img_scaled->width,
      img_scaled->height);

    wnd_info.bbox.width = static_cast<int32_t>(wnd_size_ / scale_factor + 0.5);
    wnd_info.bbox.height = wnd_info.bbox.width;

    int32_t max_x = img_scaled->width - wnd_size_;
    int32_t max_y = img_scaled->height - wnd_size_;
    for (int32_t y = 0; y <= max_y; y += slide_wnd_step_y_) {
      wnd.y = y;
      for (int32_t x = 0; x <= max_x; x += slide_wnd_step_x_) {
        wnd.x = x;
        feat_map_1->SetROI(wnd);

        wnd_info.bbox.x = static_cast<int32_t>(x / scale_factor + 0.5);
        wnd_info.bbox.y = static_cast<int32_t>(y / scale_factor + 0.5);

        for (int32_t i = 0; i < hierarchy_size_[0]; i++) {
          if (model_[i]->Classify(&score)) {
            wnd_info.score = static_cast<double>(score);
            proposals[i].push_back(wnd_info);
          }
        }
      }
    }

    img_scaled = img_pyramid->GetNextScaleImage(&scale_factor);
  }

  std::vector<std::vector<seeta::FaceInfo> > proposals_nms(hierarchy_size_[0]);
  for (int32_t i = 0; i < hierarchy_size_[0]; i++) {
      seeta::fd::NonMaximumSuppression(&(proposals[i]),  // first argument is overwritten after
      &(proposals_nms[i]), 0.8f);
    proposals[i].clear();
  }

  // Following classifiers

  seeta::ImageData img = img_pyramid->image1x();
  seeta::Rect roi;
  std::vector<float> mlp_predicts(4);  // @todo no hard-coded number!
  roi.x = roi.y = 0;
  roi.width = roi.height = wnd_size_;

  int32_t cls_idx = hierarchy_size_[0];
  int32_t model_idx = hierarchy_size_[0];
  std::vector<int32_t> buf_idx;

  for (int32_t i = 1; i < num_hierarchy_; i++) {
    buf_idx.resize(hierarchy_size_[i]);
    for (int32_t j = 0; j < hierarchy_size_[i]; j++) {
      int32_t num_wnd_src = static_cast<int32_t>(wnd_src_id_[cls_idx].size());
      std::vector<int32_t> & wnd_src = wnd_src_id_[cls_idx];
      buf_idx[j] = wnd_src[0];
      proposals[buf_idx[j]].clear();
      for (int32_t k = 0; k < num_wnd_src; k++) {
        proposals[buf_idx[j]].insert(proposals[buf_idx[j]].end(),
          proposals_nms[wnd_src[k]].begin(), proposals_nms[wnd_src[k]].end());
      }

      std::shared_ptr<seeta::fd::FeatureMap> & feat_map =
        feat_map_[cls2feat_idx_[model_[model_idx]->type()]];
      for (int32_t k = 0; k < num_stage_[cls_idx]; k++) {
        int32_t num_wnd = static_cast<int32_t>(proposals[buf_idx[j]].size());
        std::vector<seeta::FaceInfo> & bboxes = proposals[buf_idx[j]];
        int32_t bbox_idx = 0;

        for (int32_t m = 0; m < num_wnd; m++) {
          if (bboxes[m].bbox.x + bboxes[m].bbox.width <= 0 ||
              bboxes[m].bbox.y + bboxes[m].bbox.height <= 0)
            continue;
          GetWindowData(img, bboxes[m].bbox);
          feat_map->Compute(wnd_data_.data(), wnd_size_, wnd_size_);
          feat_map->SetROI(roi);

          if (model_[model_idx]->Classify(&score, mlp_predicts.data())) {
            float x = static_cast<float>(bboxes[m].bbox.x);
            float y = static_cast<float>(bboxes[m].bbox.y);
            float w = static_cast<float>(bboxes[m].bbox.width);
            float h = static_cast<float>(bboxes[m].bbox.height);

            bboxes[bbox_idx].bbox.width =
              static_cast<int32_t>((mlp_predicts[3] * 2 - 1) * w + w + 0.5);
            bboxes[bbox_idx].bbox.height = bboxes[bbox_idx].bbox.width;
            bboxes[bbox_idx].bbox.x =
              static_cast<int32_t>((mlp_predicts[1] * 2 - 1) * w + x +
              (w - bboxes[bbox_idx].bbox.width) * 0.5 + 0.5);
            bboxes[bbox_idx].bbox.y =
              static_cast<int32_t>((mlp_predicts[2] * 2 - 1) * h + y +
              (h - bboxes[bbox_idx].bbox.height) * 0.5 + 0.5);
            bboxes[bbox_idx].score = score;
            bbox_idx++;
          }
        }
        proposals[buf_idx[j]].resize(bbox_idx);

        if (k < num_stage_[cls_idx] - 1) {
            seeta::fd::NonMaximumSuppression(&(proposals[buf_idx[j]]),  // first argument overwritten after
            &(proposals_nms[buf_idx[j]]), 0.8f);
          proposals[buf_idx[j]] = proposals_nms[buf_idx[j]];
        } else {
          if (i == num_hierarchy_ - 1) {
            seeta::fd::NonMaximumSuppression(&(proposals[buf_idx[j]]),  // same here
              &(proposals_nms[buf_idx[j]]), 0.3f);
            proposals[buf_idx[j]] = proposals_nms[buf_idx[j]];
          }
        }
        model_idx++;
      }

      cls_idx++;
    }

    for (int32_t j = 0; j < hierarchy_size_[i]; j++)
      proposals_nms[j] = proposals[buf_idx[j]];
  }

  return proposals_nms[0];
}

std::shared_ptr<seeta::fd::ModelReader>
FuStDetector::CreateModelReader(seeta::fd::ClassifierType type) {
  std::shared_ptr<seeta::fd::ModelReader> reader;
  switch (type) {
  case seeta::fd::ClassifierType::LAB_Boosted_Classifier:
    reader.reset(new seeta::fd::LABBoostModelReader());
    break;
  case seeta::fd::ClassifierType::SURF_MLP:
    reader.reset(new seeta::fd::SURFMLPModelReader());
    break;
  default:
    break;
  }
  return reader;
}

std::shared_ptr<seeta::fd::Classifier>
FuStDetector::CreateClassifier(seeta::fd::ClassifierType type) {
  std::shared_ptr<seeta::fd::Classifier> classifier;
  switch (type) {
  case seeta::fd::ClassifierType::LAB_Boosted_Classifier:
    classifier.reset(new seeta::fd::LABBoostedClassifier());
    break;
  case seeta::fd::ClassifierType::SURF_MLP:
    classifier.reset(new seeta::fd::SURFMLP());
    break;
  default:
    break;
  }
  return classifier;
}

std::shared_ptr<seeta::fd::FeatureMap>
FuStDetector::CreateFeatureMap(seeta::fd::ClassifierType type) {
  std::shared_ptr<seeta::fd::FeatureMap> feat_map;
  switch (type) {
  case seeta::fd::ClassifierType::LAB_Boosted_Classifier:
    feat_map.reset(new seeta::fd::LABFeatureMap());
    break;
  case seeta::fd::ClassifierType::SURF_MLP:
    feat_map.reset(new seeta::fd::SURFFeatureMap());
    break;
  default:
    break;
  }
  return feat_map;
}

void FuStDetector::GetWindowData(const seeta::ImageData & img,
    const seeta::Rect & wnd) {
  int32_t pad_left;
  int32_t pad_right;
  int32_t pad_top;
  int32_t pad_bottom;
  seeta::Rect roi = wnd;

  pad_left = pad_right = pad_top = pad_bottom = 0;
  if (roi.x + roi.width > img.width)
    pad_right = roi.x + roi.width - img.width;
  if (roi.x < 0) {
    pad_left = -roi.x;
    roi.x = 0;
  }
  if (roi.y + roi.height > img.height)
    pad_bottom = roi.y + roi.height - img.height;
  if (roi.y < 0) {
    pad_top = -roi.y;
    roi.y = 0;
  }

  wnd_data_buf_.resize(roi.width * roi.height);
  const uint8_t* src = img.data + roi.y * img.width + roi.x;
  uint8_t* dest = wnd_data_buf_.data();
  int32_t len = sizeof(uint8_t) * roi.width;
  int32_t len2 = sizeof(uint8_t) * (roi.width - pad_left - pad_right);

  if (pad_top > 0) {
    std::memset(dest, 0, len * pad_top);
    dest += (roi.width * pad_top);
  }
  if (pad_left == 0) {
    if (pad_right == 0) {
      for (int32_t y = pad_top; y < roi.height - pad_bottom; y++) {
        std::memcpy(dest, src, len);
        src += img.width;
        dest += roi.width;
      }
    } else {
      for (int32_t y = pad_top; y < roi.height - pad_bottom; y++) {
        std::memcpy(dest, src, len2);
        src += img.width;
        dest += roi.width;
        std::memset(dest - pad_right, 0, sizeof(uint8_t) * pad_right);
      }
    }
  } else {
    if (pad_right == 0) {
      for (int32_t y = pad_top; y < roi.height - pad_bottom; y++) {
        std::memset(dest, 0, sizeof(uint8_t)* pad_left);
        std::memcpy(dest + pad_left, src, len2);
        src += img.width;
        dest += roi.width;
      }
    } else {
      for (int32_t y = pad_top; y < roi.height - pad_bottom; y++) {
        std::memset(dest, 0, sizeof(uint8_t) * pad_left);
        std::memcpy(dest + pad_left, src, len2);
        src += img.width;
        dest += roi.width;
        std::memset(dest - pad_right, 0, sizeof(uint8_t) * pad_right);
      }
    }
  }
  if (pad_bottom > 0)
    std::memset(dest, 0, len * pad_bottom);

  seeta::ImageData src_img(roi.width, roi.height);
  seeta::ImageData dest_img(wnd_size_, wnd_size_);
  src_img.data = wnd_data_buf_.data();
  dest_img.data = wnd_data_.data();
  seeta::fd::ResizeImage(src_img, &dest_img);
}

}  // namespace fd


// src/face_detection.cpp

class FaceDetection::Impl {
 public:
  Impl()
      : detector_(new seeta::fd::FuStDetector()),
        slide_wnd_step_x_(4), slide_wnd_step_y_(4),
        min_face_size_(20), max_face_size_(-1),
        cls_thresh_(3.85f) {}

  ~Impl() {}

  inline bool IsLegalImage(const seeta::ImageData & image) {
    return (image.num_channels == 1 && image.width > 0 && image.height > 0 &&
      image.data != nullptr);
  }

 public:
  static const int32_t kWndSize = 40;

  int32_t min_face_size_;
  int32_t max_face_size_;
  int32_t slide_wnd_step_x_;
  int32_t slide_wnd_step_y_;
  float cls_thresh_;

  std::vector<seeta::FaceInfo> pos_wnds_;
  std::unique_ptr<seeta::fd::Detector> detector_;
  seeta::fd::ImagePyramid img_pyramid_;
};

FaceDetection::FaceDetection(const char* model_path)
    : impl_(new seeta::FaceDetection::Impl()) {
  impl_->detector_->LoadModel(model_path);
}

FaceDetection::~FaceDetection() {
  if (impl_ != nullptr)
    delete impl_;
}

std::vector<seeta::FaceInfo> FaceDetection::Detect(
    const seeta::ImageData & img) {
  if (!impl_->IsLegalImage(img))
    return std::vector<seeta::FaceInfo>();

  int32_t min_img_size = img.height <= img.width ? img.height : img.width;
  min_img_size = (impl_->max_face_size_ > 0 ?
    (min_img_size >= impl_->max_face_size_ ? impl_->max_face_size_ : min_img_size) :
    min_img_size);

  impl_->img_pyramid_.SetImage1x(img.data, img.width, img.height);
  impl_->img_pyramid_.SetMinScale(static_cast<float>(impl_->kWndSize) / min_img_size);

  impl_->detector_->SetWindowSize(impl_->kWndSize);
  impl_->detector_->SetSlideWindowStep(impl_->slide_wnd_step_x_,
    impl_->slide_wnd_step_y_);

  impl_->pos_wnds_ = impl_->detector_->Detect(&(impl_->img_pyramid_));

  for (int32_t i = 0; i < impl_->pos_wnds_.size(); i++) {
    if (impl_->pos_wnds_[i].score < impl_->cls_thresh_) {
      impl_->pos_wnds_.resize(i);
      break;
    }
  }

  return impl_->pos_wnds_;
}

void FaceDetection::SetMinFaceSize(int32_t size) {
  if (size >= 20) {
    impl_->min_face_size_ = size;
    impl_->img_pyramid_.SetMaxScale(impl_->kWndSize / static_cast<float>(size));
  }
}

void FaceDetection::SetMaxFaceSize(int32_t size) {
  if (size >= 0)
    impl_->max_face_size_ = size;
}

void FaceDetection::SetImagePyramidScaleFactor(float factor) {
  if (factor >= 0.01f && factor <= 0.99f)
    impl_->img_pyramid_.SetScaleStep(static_cast<float>(factor));
}

void FaceDetection::SetWindowStep(int32_t step_x, int32_t step_y) {
  if (step_x > 0)
    impl_->slide_wnd_step_x_ = step_x;
  if (step_y > 0)
    impl_->slide_wnd_step_y_ = step_y;
}

void FaceDetection::SetScoreThresh(float thresh) {
  if (thresh >= 0)
    impl_->cls_thresh_ = thresh;
}

}  // namespace seeta
 */
