"""
弹道与子弹物理系统 - 核心模块

物理模型：
- 重力：y 轴负方向，g = 9.81 m/s^2
- 空气阻力：二次阻力模型 a = -k * v^2，k 为武器阻力系数（1/m）
  水平方向有解析解：v(t) = v0 / (1 + k*v0*t)，x(t) = ln(1 + k*v0*t) / k
- 风向约定：wind_direction 为风向与子弹飞行方向的夹角（度），
  0 = 顺风，180 = 逆风，90/270 = 横风
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math
import time

GRAVITY = 9.81

# 武器档案：初速、阻力系数、最大射程、穿透能力
WEAPON_PROFILES: Dict[str, Dict[str, float]] = {
    "pistol":  {"muzzle_velocity": 350.0, "drag_coefficient": 0.0015, "max_range": 200.0,  "penetration_power": 60.0},
    "smg":     {"muzzle_velocity": 400.0, "drag_coefficient": 0.0014, "max_range": 300.0,  "penetration_power": 80.0},
    "rifle":   {"muzzle_velocity": 880.0, "drag_coefficient": 0.0006, "max_range": 1000.0, "penetration_power": 200.0},
    "sniper":  {"muzzle_velocity": 920.0, "drag_coefficient": 0.0004, "max_range": 1500.0, "penetration_power": 320.0},
    "shotgun": {"muzzle_velocity": 380.0, "drag_coefficient": 0.0020, "max_range": 80.0,   "penetration_power": 40.0},
}

# 材质档案：每米厚度的穿透阻力、穿透后的伤害保留率
MATERIALS: Dict[str, Dict[str, float]] = {
    "wood":     {"resistance": 40.0,  "damage_retention": 0.7},
    "concrete": {"resistance": 150.0, "damage_retention": 0.4},
    "steel":    {"resistance": 800.0, "damage_retention": 0.2},
}

# 风对子弹的耦合系数与参考弹速（用于体现低速弹受风影响更大）
WIND_COUPLING = 0.1
REFERENCE_BULLET_SPEED = 300.0


@dataclass
class Config:
    pass


@dataclass
class Bullet:
    """场景中的子弹对象，由对象池复用。"""
    position: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    velocity: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    weapon_type: str = "rifle"
    speed: float = 0.0
    max_range: float = 0.0
    age: float = 0.0
    traveled: float = 0.0
    active: bool = False

    @property
    def lifetime(self) -> float:
        """生命周期按最大射程与初速计算，而非写死的固定值。"""
        if self.speed <= 0:
            return 0.0
        return self.max_range / self.speed

    @property
    def remaining_life(self) -> float:
        return max(0.0, self.lifetime - self.age)


class BulletPhysics:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self._state = {}
        self._history = []
        self._active: List[Bullet] = []
        self._pool: List[Bullet] = []
        self.weapon_profiles = dict(WEAPON_PROFILES)
        self.weapon_profiles.update(self.config.get("weapon_profiles", {}))
        self.materials = dict(MATERIALS)
        self.materials.update(self.config.get("materials", {}))

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------
    def _profile(self, weapon_type: str) -> Dict[str, float]:
        if weapon_type not in self.weapon_profiles:
            raise ValueError(f"未知武器类型: {weapon_type}")
        return self.weapon_profiles[weapon_type]

    @staticmethod
    def _normalize(direction: Tuple[float, float, float]) -> Tuple[float, float, float]:
        length = math.sqrt(sum(c * c for c in direction))
        if length == 0:
            raise ValueError("方向向量不能为零向量")
        return (direction[0] / length, direction[1] / length, direction[2] / length)

    # ------------------------------------------------------------------
    # 弹道计算：重力 + 飞行时间 + 空气阻力
    # ------------------------------------------------------------------
    def flight_time(self, distance: float, weapon_type: str = "rifle") -> float:
        """子弹飞行指定水平距离所需的时间（二次阻力模型解析解）。"""
        if distance < 0:
            raise ValueError("距离不能为负")
        profile = self._profile(weapon_type)
        v0 = profile["muzzle_velocity"]
        k = profile["drag_coefficient"]
        # x(t) = ln(1 + k*v0*t) / k  =>  t = (exp(k*x) - 1) / (k*v0)
        return math.expm1(k * distance) / (k * v0)

    def calculate_trajectory(
        self,
        origin: Tuple[float, float, float],
        direction: Tuple[float, float, float],
        weapon_type: str = "rifle",
        distance: Optional[float] = None,
        dt: float = 0.005,
    ) -> List[Dict[str, float]]:
        """
        数值积分计算弹道，考虑重力与空气阻力。
        返回弹道采样点列表，每点包含 x/y/z/t/speed。
        """
        profile = self._profile(weapon_type)
        k = profile["drag_coefficient"]
        max_range = distance if distance is not None else profile["max_range"]
        dir_x, dir_y, dir_z = self._normalize(direction)

        pos = [float(origin[0]), float(origin[1]), float(origin[2])]
        vel = [dir_x * profile["muzzle_velocity"],
               dir_y * profile["muzzle_velocity"],
               dir_z * profile["muzzle_velocity"]]

        points = [{
            "x": pos[0], "y": pos[1], "z": pos[2], "t": 0.0,
            "speed": math.sqrt(sum(c * c for c in vel)),
        }]
        t = 0.0
        while t < 600.0:
            speed = math.sqrt(vel[0] ** 2 + vel[1] ** 2 + vel[2] ** 2)
            if speed <= 0:
                break
            drag = k * speed  # 二次阻力：a = -k*|v|*v
            vel[0] -= drag * vel[0] * dt
            vel[1] -= (drag * vel[1] + GRAVITY) * dt
            vel[2] -= drag * vel[2] * dt
            pos[0] += vel[0] * dt
            pos[1] += vel[1] * dt
            pos[2] += vel[2] * dt
            t += dt
            points.append({
                "x": pos[0], "y": pos[1], "z": pos[2], "t": t,
                "speed": math.sqrt(vel[0] ** 2 + vel[1] ** 2 + vel[2] ** 2),
            })
            horizontal = math.hypot(pos[0] - origin[0], pos[2] - origin[2])
            if horizontal >= max_range:
                break
        return points

    # ------------------------------------------------------------------
    # 子弹下坠：按武器类型与距离计算
    # ------------------------------------------------------------------
    def bullet_drop(self, distance: float, weapon_type: str = "rifle") -> float:
        """水平射击时子弹在指定距离处的下坠量（米）。"""
        t = self.flight_time(distance, weapon_type)
        return 0.5 * GRAVITY * t * t

    # ------------------------------------------------------------------
    # 风速影响：考虑风向与子弹速度
    # ------------------------------------------------------------------
    def wind_effect(
        self,
        wind_speed: float,
        wind_direction: float,
        bullet_speed: float,
        flight_time: float,
    ) -> Dict[str, float]:
        """
        计算风对子弹的影响。
        wind_direction: 风向与弹道的夹角（度），0=顺风，180=逆风，90/270=横风。
        返回横向偏移、顺/逆风速度修正与飞行时间修正。
        """
        if bullet_speed <= 0:
            raise ValueError("子弹速度必须为正")
        rad = math.radians(wind_direction)
        cross = wind_speed * math.sin(rad)   # 横风分量
        along = wind_speed * math.cos(rad)   # 顺/逆风分量
        # 低速子弹受风影响更大
        speed_factor = REFERENCE_BULLET_SPEED / bullet_speed
        lateral_drift = 0.5 * WIND_COUPLING * cross * flight_time ** 2 * speed_factor
        speed_delta = WIND_COUPLING * along * flight_time * speed_factor
        time_delta = 0.0
        if bullet_speed + speed_delta > 0:
            time_delta = flight_time * speed_delta / (bullet_speed + speed_delta)
        return {
            "lateral_drift": lateral_drift,
            "speed_delta": speed_delta,
            "time_delta": time_delta,
        }

    # ------------------------------------------------------------------
    # 穿透：伤害衰减 + 材质 + 厚度
    # ------------------------------------------------------------------
    def penetration(
        self,
        material: str,
        thickness: float,
        damage: float,
        weapon_type: str = "rifle",
    ) -> Dict[str, float]:
        """
        计算子弹穿透材质后的结果。
        返回能否穿透、穿透后的伤害（含材质与厚度衰减）。
        """
        if material not in self.materials:
            raise ValueError(f"未知材质: {material}")
        if thickness < 0:
            raise ValueError("厚度不能为负")
        mat = self.materials[material]
        power = self._profile(weapon_type)["penetration_power"]
        resistance = mat["resistance"] * thickness
        can_penetrated = power > resistance
        if can_penetrated:
            damage_after = damage * mat["damage_retention"] * (1.0 - resistance / power)
        else:
            damage_after = 0.0
        return {
            "can_penetrated": can_penetrated,
            "damage_after": damage_after,
            "resistance": resistance,
        }

    # ------------------------------------------------------------------
    # 子弹生命周期：按距离/速度计算 + 过期消失 + 对象池
    # ------------------------------------------------------------------
    def bullet_life(self, weapon_type: str = "rifle", max_range: Optional[float] = None,
                    speed: Optional[float] = None) -> float:
        """按射程与弹速计算子弹生命周期（秒），不再是写死的固定值。"""
        profile = self._profile(weapon_type)
        rng = max_range if max_range is not None else profile["max_range"]
        spd = speed if speed is not None else profile["muzzle_velocity"]
        if spd <= 0:
            raise ValueError("子弹速度必须为正")
        return rng / spd

    def spawn_bullet(
        self,
        origin: Tuple[float, float, float],
        direction: Tuple[float, float, float],
        weapon_type: str = "rifle",
        speed: Optional[float] = None,
        max_range: Optional[float] = None,
    ) -> Bullet:
        """从对象池获取（或新建）子弹对象并发射。"""
        profile = self._profile(weapon_type)
        spd = speed if speed is not None else profile["muzzle_velocity"]
        rng = max_range if max_range is not None else profile["max_range"]
        dir_x, dir_y, dir_z = self._normalize(direction)

        bullet = self._pool.pop() if self._pool else Bullet()
        bullet.position = [float(origin[0]), float(origin[1]), float(origin[2])]
        bullet.velocity = [dir_x * spd, dir_y * spd, dir_z * spd]
        bullet.weapon_type = weapon_type
        bullet.speed = spd
        bullet.max_range = rng
        bullet.age = 0.0
        bullet.traveled = 0.0
        bullet.active = True
        self._active.append(bullet)
        return bullet

    @property
    def active_bullets(self) -> List[Bullet]:
        return list(self._active)

    @property
    def pool_size(self) -> int:
        return len(self._pool)

    def update(self, dt: float):
        """推进场景中所有活跃子弹，过期子弹从场景移除并回收进对象池。"""
        if dt < 0:
            raise ValueError("dt 不能为负")
        still_active = []
        for bullet in self._active:
            profile = self._profile(bullet.weapon_type)
            k = profile["drag_coefficient"]
            speed = math.sqrt(sum(c * c for c in bullet.velocity))
            if speed > 0:
                drag = k * speed
                bullet.velocity[0] -= drag * bullet.velocity[0] * dt
                bullet.velocity[1] -= (drag * bullet.velocity[1] + GRAVITY) * dt
                bullet.velocity[2] -= drag * bullet.velocity[2] * dt
                step = [c * dt for c in bullet.velocity]
                bullet.position[0] += step[0]
                bullet.position[1] += step[1]
                bullet.position[2] += step[2]
                bullet.traveled += math.sqrt(sum(c * c for c in step))
            bullet.age += dt
            if bullet.age >= bullet.lifetime or bullet.traveled >= bullet.max_range:
                bullet.active = False
                self._history.append(bullet)
                self._pool.append(bullet)  # 回收进对象池复用
            else:
                still_active.append(bullet)
        self._active = still_active

    def reset(self):
        self._state = {}
        self._history = []
        self._active = []
        self._pool = []
