"""
业务端告警 Pipeline：基于带属性检测结果，判断违规并触发告警。

违规条件：
  - 未佩戴安全帽
  - 未穿指定颜色工装马甲，且在安全区域内
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import random
import time


@dataclass
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass
class Detection:
    """
    单个检测框结果。

    兼容“属性检测网络”：
    - class_id / class_name：主类别（如 person、helmet、vest）
    - track_id：跨帧追踪 ID
    - parent_track_id：当检测框是 person 的从属属性目标时，指向 person 的 track_id
    - attributes：属性字典（例如 {"helmet_color": "yellow", "vest_color": "orange", "has_vest": True}）
    """

    bbox: BoundingBox
    class_id: int
    class_name: str
    track_id: int
    parent_track_id: Optional[int] = None
    attributes: Dict[str, object] = field(default_factory=dict)


@dataclass
class FrameDetectionResult:
    """
    模型输出接口定义。

    字段覆盖题目要求：
    - timestamp: 检测时间戳
    - frame_id: 检测帧图片 ID
    - camera_id: 检测相机 ID
    - detections: 检测框结果（含位置、类别、追踪 ID）
    - class_name_map: 类别 ID 到名称映射
    """

    timestamp: float
    frame_id: str
    camera_id: str
    detections: List[Detection] = field(default_factory=list)
    class_name_map: Dict[int, str] = field(default_factory=dict)


def is_in_safety_zone(
    worker: Detection,
    camera_id: str,
    zone_polygon: List[Tuple[float, float]],
    *,
    mode: str = "time_segment",
    now_ts: Optional[float] = None,
) -> bool:
    """
    Mock：判断工人是否在安全区域内。

    输入：
      - worker: 工人检测结果（含位置、追踪 ID）
      - camera_id: 相机 ID
      - zone_polygon: 多边形顶点列表（首尾相连）
    输出：
      - bool: 是否在安全区域内

    mock 模式：
      - mode="fixed_true"  : 恒 True
      - mode="fixed_false" : 恒 False
      - mode="random"      : 随机 True/False
      - mode="time_segment": 基于时间片段连续变化，模拟“进/出区域”
    """
    if mode == "fixed_true":
        return True
    if mode == "fixed_false":
        return False
    if mode == "random":
        return random.random() < 0.5

    # 默认：time_segment（连续区段）
    ts = now_ts if now_ts is not None else time.time()

    # 使用 camera_id + track_id 形成不同 worker 的相位偏移
    worker_key = sum(ord(c) for c in camera_id) + worker.track_id * 17
    # 每 8 秒一个完整周期：前 4 秒在区内，后 4 秒在区外
    phase = int(ts + worker_key) % 8
    return phase < 4


class AlertPipeline:
    ALERT_LATENCY_SEC = 1.0
    COOLDOWN_SEC = 5.0

    def __init__(
        self,
        required_vest_color: str = "orange",
        safety_zone_mode: str = "time_segment",
    ):
        self.required_vest_color = required_vest_color.lower()
        self.safety_zone_mode = safety_zone_mode

        self._cooldown_until: Optional[float] = None
        self._active_violations: set[Tuple[int, str]] = set()
        self._first_seen_violation_ts: Dict[Tuple[int, str], float] = {}

    def _build_person_attr_index(self, result: FrameDetectionResult) -> Dict[int, Dict[str, object]]:
        """
        以 person track_id 为键，聚合属性。

        兼容两种输入：
        1) person 自身 attributes 已包含 has_helmet/vest_color 等
        2) helmet/vest 是独立检测框，通过 parent_track_id 关联 person
        """
        person_index: Dict[int, Dict[str, object]] = {}

        # 先初始化 person
        for det in result.detections:
            if det.class_name.lower() == "person":
                person_index[det.track_id] = {
                    "person_det": det,
                    "has_helmet": bool(det.attributes.get("has_helmet", False)),
                    "helmet_color": det.attributes.get("helmet_color"),
                    "has_vest": bool(det.attributes.get("has_vest", False)),
                    "vest_color": (det.attributes.get("vest_color") or "").lower() if det.attributes.get("vest_color") else None,
                    "clothes_color": det.attributes.get("clothes_color"),
                }

        # 再用从属检测框补齐
        for det in result.detections:
            cname = det.class_name.lower()
            if cname.startswith("helmet") and det.parent_track_id in person_index:
                entry = person_index[det.parent_track_id]
                entry["has_helmet"] = True
                c = det.attributes.get("color") or det.attributes.get("helmet_color")
                if c:
                    entry["helmet_color"] = c
            elif cname.startswith("vest") and det.parent_track_id in person_index:
                entry = person_index[det.parent_track_id]
                entry["has_vest"] = True
                c = det.attributes.get("color") or det.attributes.get("vest_color")
                if c:
                    entry["vest_color"] = str(c).lower()

        return person_index

    def _collect_violations(
        self,
        result: FrameDetectionResult,
        zone_polygon: List[Tuple[float, float]],
    ) -> set[Tuple[int, str]]:
        """收集当前帧违规集合，元素格式：(person_track_id, violation_type)"""
        violations: set[Tuple[int, str]] = set()
        person_index = self._build_person_attr_index(result)

        for person_id, info in person_index.items():
            person_det: Detection = info["person_det"]
            has_helmet = bool(info["has_helmet"])
            has_vest = bool(info["has_vest"])
            vest_color = info["vest_color"]

            # 违规1：未佩戴安全帽（不依赖区域）
            if not has_helmet:
                violations.add((person_id, "no_helmet"))

            # 违规2：在安全区域内，未穿指定颜色工装马甲
            in_zone = is_in_safety_zone(
                person_det,
                result.camera_id,
                zone_polygon,
                mode=self.safety_zone_mode,
                now_ts=result.timestamp,
            )
            if in_zone:
                wrong_vest = (not has_vest) or (vest_color != self.required_vest_color)
                if wrong_vest:
                    violations.add((person_id, "wrong_vest_in_zone"))

        return violations

    def process_frame(self, result: FrameDetectionResult, zone_polygon: List[Tuple[float, float]]) -> bool:
        """
        处理单帧检测结果，返回本帧是否应发出告警。

        规则：
        - 首帧违规到告警：<= 1 秒（本实现可做到同帧触发）
        - 持续违规：持续报警（每帧 True）
        - 事件消除后：进入 5 秒 cooldown，期间无新危险不重复报警
        - cooldown 内若出现“新的危险事件”（新 person 或新 violation_type），立即恢复报警
        """
        now = result.timestamp
        current_violations = self._collect_violations(result, zone_polygon)

        # 记录首次出现时间（用于审计/扩展）
        for v in current_violations:
            self._first_seen_violation_ts.setdefault(v, now)

        # 状态切换：有违规 -> 持续报警
        if current_violations:
            in_cooldown = self._cooldown_until is not None and now < self._cooldown_until

            # cooldown 期间：统一抑制，直到 cooldown 结束
            if in_cooldown:
                self._active_violations = current_violations
                return False

            # 非 cooldown：有违规就报警（满足 <=1s，通常同帧）
            self._active_violations = current_violations
            return True

        # 无违规：若刚从“有违规”转为“无违规”，启动 cooldown
        if self._active_violations:
            self._cooldown_until = now + self.COOLDOWN_SEC

        self._active_violations = set()

        # 清理长期不用的首次出现记录（仅保留当前活跃）
        if not current_violations:
            self._first_seen_violation_ts.clear()

        return False
