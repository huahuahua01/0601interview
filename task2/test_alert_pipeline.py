from task2.alert_pipeline import (
    AlertPipeline,
    BoundingBox,
    Detection,
    FrameDetectionResult,
)


def _person(track_id: int = 1):
    return Detection(
        bbox=BoundingBox(10, 10, 60, 120),
        class_id=0,
        class_name="person",
        track_id=track_id,
    )


def _helmet(track_id: int = 1, color: str = "yellow"):
    return Detection(
        bbox=BoundingBox(20, 5, 50, 30),
        class_id=1,
        class_name=f"helmet_{color}",
        track_id=track_id,
        parent_track_id=track_id,
        attributes={"color": color},
    )


def _vest(track_id: int = 1, color: str = "orange"):
    return Detection(
        bbox=BoundingBox(20, 40, 55, 95),
        class_id=2,
        class_name=f"vest_{color}",
        track_id=track_id,
        parent_track_id=track_id,
        attributes={"color": color},
    )


def test_alert_trigger_continuous_and_cooldown():
    """演示：告警触发、持续报警、Cooldown 抑制。"""
    pipeline = AlertPipeline(required_vest_color="orange", safety_zone_mode="fixed_true")
    zone_polygon = [(0, 0), (200, 0), (200, 200), (0, 200)]
    t0 = 1000.0

    # 1) 首次违规：无 helmet，应立即告警
    f1 = FrameDetectionResult(
        timestamp=t0,
        frame_id="f1",
        camera_id="cam_1",
        detections=[_person(1), _vest(1, "orange")],
        class_name_map={0: "person", 1: "helmet", 2: "vest"},
    )
    d1 = pipeline.process_frame(f1, zone_polygon)
    assert d1 is True

    # 2) 持续违规：继续无 helmet，继续告警
    f2 = FrameDetectionResult(
        timestamp=t0 + 0.5,
        frame_id="f2",
        camera_id="cam_1",
        detections=[_person(1), _vest(1, "orange")],
    )
    d2 = pipeline.process_frame(f2, zone_polygon)
    assert d2 is True

    # 3) 违规消除：补齐 helmet 和正确 vest，不告警
    f3 = FrameDetectionResult(
        timestamp=t0 + 1.0,
        frame_id="f3",
        camera_id="cam_1",
        detections=[_person(1), _helmet(1, "yellow"), _vest(1, "orange")],
    )
    d3 = pipeline.process_frame(f3, zone_polygon)
    assert d3 is False

    # 4) Cooldown 内重新违规：抑制重复报警
    f4 = FrameDetectionResult(
        timestamp=t0 + 2.0,
        frame_id="f4",
        camera_id="cam_1",
        detections=[_person(1), _helmet(1, "yellow"), _vest(1, "blue")],
    )
    d4 = pipeline.process_frame(f4, zone_polygon)
    assert d4 is False

    # 5) Cooldown 结束后再次违规：恢复告警
    f5 = FrameDetectionResult(
        timestamp=t0 + 7.0,
        frame_id="f5",
        camera_id="cam_1",
        detections=[_person(1), _helmet(1, "yellow"), _vest(1, "blue")],
    )
    d5 = pipeline.process_frame(f5, zone_polygon)
    assert d5 is True
