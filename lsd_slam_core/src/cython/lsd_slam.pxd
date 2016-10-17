from eigency.core cimport *
from libcpp.deque cimport deque
from libcpp.memory cimport shared_ptr
from libcpp cimport bool

cdef extern from "../DepthEstimation/DepthMapPixelHypothesis.h" namespace "lsd_slam":
    cdef cppclass DepthMapPixelHypothesis:
        bool isValid
        # Flag that blacklists a point to never be used - set if stereo fails repeatedly on this pixel.
        int blacklisted
        # How many frames to skip ahead in the tracked-frames-queue.
        float nextStereoFrameMinID
        # Counter for validity, basically how many successful observations are incorporated.
        int validity_counter
        # Actual Gaussian Distribution.
        float idepth
        float idepth_var
        # Smoothed Gaussian Distribution.
        float idepth_smoothed
        float idepth_var_smoothed

cdef extern from "../DataStructures/Frame.h" namespace "lsd_slam":
    cdef cppclass Frame:
        Frame(int id, int width, int height, FlattenedMap[Matrix, float, _3, _3] &K, double timestamp, const unsigned char* image)
#        _Frame "Frame"(int id, int width, int height, FlattenedMap[Matrix, float, _3, _3] &K, double timestamp, const float* image)
        DepthMapPixelHypothesis* otherDepthMap
        DepthMapPixelHypothesis* currentDepthMap
        const float* idepth(int level)
        const float* idepthVar(int level)
        float* image(int level)


cdef extern from "../DepthEstimation/DepthMap.h" namespace "lsd_slam" nogil:
    cdef cppclass DepthMap:
        DepthMap(int w, int h, FlattenedMap[Matrix, float, _3, _3] &K)
        void updateKeyframe(deque[ shared_ptr[Frame] ] referenceFrames)
        void createKeyFrame(Frame* new_keyframe)
        void finalizeKeyFrame()
        void initializeRandomly(Frame* new_frame)
        void setFromExistingKF(Frame* kf)

        void reset()


