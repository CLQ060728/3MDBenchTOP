from .video_keyframe_detector.KeyFrameDetector.key_frame_detector import keyframeDetection
from Katna.video import Video
from Katna.writer import KeyFrameDiskWriter


def compute_peak_est_key_frames(vid_file_path, output_path, diff_threshold):
    keyframeDetection(vid_file_path, output_path, float(diff_threshold),
                      plotMetrics=True, verbose=True)


def compute_katna_key_frames(vid_file_path, output_path):
    vid = Video()
    no_frames = 200  # decided by itself, according to different videos
    disk_writer = KeyFrameDiskWriter(location=output_path)
    vid.extract_video_keyframes(no_of_frames=no_frames, file_path=vid_file_path, writer=disk_writer)