from eigency.core cimport *
from libcpp.deque cimport deque
from libcpp.vector cimport vector
from libcpp.memory cimport shared_ptr
from libcpp cimport bool

cdef extern from "../DepthEstimation/DepthMapPixelHypothesis.h" namespace "lsd_slam":
    cppclass DepthMapPixelHypothesis:
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


cdef extern from "../DataStructures/FramePoseStruct.h" namespace "lsd_slam" nogil:

    cppclass FramePoseStruct:
        FramePoseStruct(Frame* frame)
        FramePoseStruct* trackingParent
        Sim3 thisToParent_raw
        Frame* frame
        Sim3 getCamToWorld()
        int frameID

    cppclass pFramePoseStruct_aligned_allocator "Eigen::aligned_allocator<lsd_slam::FramePoseStruct*>":
            pass
ctypedef FramePoseStruct* pFramePoseStruct

cdef extern from "../DataStructures/Frame.h" namespace "lsd_slam":
    cppclass Frame:
        Frame(int id, int width, int height, Map[Matrix3f] &K, double timestamp, const unsigned char* image)
#        _Frame "Frame"(int id, int width, int height, FlattenedMap[Matrix, float, _3, _3] &K, double timestamp, const float* image)
        DepthMapPixelHypothesis* otherDepthMap
        DepthMapPixelHypothesis* currentDepthMap
        const float* idepth(int level)
        const float* idepthVar(int level)
        float* image(int level)
        FramePoseStruct* pose
        Sim3 getScaledCamToWorld()
        int width()
        int height()
    cppclass pFrame_aligned_allocator "Eigen::aligned_allocator<lsd_slam::Frame*>":
        pass
ctypedef Frame* pFrame

cdef extern from "../GlobalMapping/KeyFrameGraph.h" namespace "lsd_slam" nogil:

    cppclass KeyFrameGraph:
        void addKeyFrame(Frame* frame)
        void addFrame(Frame* frame)
        vector[ pFrame, pFrame_aligned_allocator] keyframesAll

cdef extern from "../DepthEstimation/DepthMap.h" namespace "lsd_slam" nogil:
    cppclass DepthMap:
        DepthMap(int w, int h, Map[Matrix3f] &K)
        void updateKeyframe(deque[ shared_ptr[Frame] ] referenceFrames)
        void createKeyFrame(Frame* new_keyframe)
        void finalizeKeyFrame()
        void initializeRandomly(Frame* new_frame)
        void setFromExistingKF(Frame* kf)
        void reset()

cdef extern from "../util/SophusUtil.h":
    cppclass SE3:
        Matrix4d matrix()

    cppclass Sim3:
        Sim3(Map[Matrix4d]& T)
        Matrix3d rotationMatrix()  # const Matrix<Scalar,3,3>
        Vector3d translation()
        Matrix4d matrix()

        void setRotationMatrix  (Map[Matrix3d] & R)
        void setScale(const double & scale)
        Sim3 inverse()
    cppclass SO3:
        pass

cdef extern from "../IOWrapper/ROS/ROSOutput3DWrapper.h" namespace "lsd_slam":
    cppclass ROSOutput3DWrapper:
        ROSOutput3DWrapper(int width, int height)

cdef extern from "../SlamSystem.h" namespace "lsd_slam" nogil:

    cppclass SlamSystem:
        SlamSystem(int w, int h, Map[Matrix3f] K, bool enableSLAM)
        void randomInit(unsigned char* image, double timeStamp, int id)
        void trackFrame(unsigned char* image, unsigned int frameID, bool blockUntilMapped, double timestamp)
        void finalize()

        Frame* getCurrentKeyframe()
        SE3 getCurrentPoseEstimate()

        bool doMappingIteration()
        void optimizeGraph()
        bool optimizationIteration(int itsPerTry, float minChange)
        void setVisualization(ROSOutput3DWrapper* outputWrapper)
        void publishKeyframeGraph()
        vector[pFramePoseStruct, pFramePoseStruct_aligned_allocator] getAllPoses()
        vector[ pFrame, pFrame_aligned_allocator] getAllKeyFrames()

cdef extern from "../util/settings.h" namespace "lsd_slam":
    int debugDisplay
    bool displayDepthMap
    bool onSceenInfoDisplay

