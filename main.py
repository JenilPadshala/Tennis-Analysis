from utils import read_video, save_video
from trackers import PlayerTracker
from trackers import BallTracker


def main():
    # read video
    input_video_path = "input_videos/input_video.mp4"
    video_frames = read_video(input_video_path)

    # Detect players in the video
    player_tracker = PlayerTracker(model_path="yolov8x.pt")
    player_detections = player_tracker.detect_frames(
        video_frames,
        read_from_stub=True,
        stub_path="tracker_stubs/player_detections.pkl",
    )
    # Detect ball
    ball_tracker = BallTracker(model_path="models/yolo5_best.pt")
    ball_detections = ball_tracker.detect_frames(
        video_frames,
        read_from_stub=True,
        stub_path="tracker_stubs/ball_detections.pkl",
    )

    # Draw output

    # Draw Player Bounding Boxes
    output_video_frames = player_tracker.draw_bboxes(video_frames, player_detections)

    # Draw ball output
    output_video_frames = ball_tracker.draw_bboxes(video_frames, ball_detections)

    save_video(video_frames, "output_videos/output_video.avi")


if __name__ == "__main__":
    main()
