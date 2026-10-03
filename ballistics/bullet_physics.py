"""
弹道与子弹物理系统 - 核心模块

修复内容：
- calculate_trajectory(): 考虑重力、飞行时间与空气阻力
- bullet_drop(): 按武器类型与距离计算下坠
- wind_effect(): 考虑风向（顺风/逆风/横风）与子弹速度
- penetration(): 计算伤害衰减，考虑材质与厚度
- 子弹生命周期：按速度/射程计算、过期消失、对象池复用
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math


GRAVITY = 9.81

# 各武器类型的弹道参数
WEAPON_PROFILES: Dict[str, Dict[str, float]] = {
    "pistol":  {"muzzle_velocity": 350.0, "drag_coefficient": 0.020, "mass": 0.008, "max_range": 200.0},
    "smg":     {"muzzle_velocity": 400.0, "drag_coefficient": 0.018, "mass": 0.008, "max_range": 300.0},
    "rifle":   {"muzzle_velocity": 800.0, "drag_coefficient": 0.010, "mass": 0.010, "max_range": 800.0},
    "sniper":  {"muzzle_velocity": 900.0, "drag_coefficient": 0.008, "mass": 0.012, "max_range": 1500.0},
    "shotgun": {"muzzle_velocity": 380.0, "drag_coefficient": 0.030, "mass": 0.030, "max_range": 100.0},
}

# 材质参数：resistance 为穿透每米所需能量(J/m)，retention 为穿透后能量保留系数
MATERIALS: Dict[str, Dict[str, float]] = {
    "wood":     {"resistance": 2_000.0,  "retention": 0.85},
    "concrete": {"resistance": 8_000.0,  "retention": 0.60},
    "steel":    {"resistance": 80_000.0, "retention": 0.35},
}

DEFAULT_WEAPON = "rifle"


@dataclass
class Config:
    pass


@dataclass
class Bullet:
    """子弹对象，由对象池复用。"""
    position: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    velocity: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    weapon_type: str = DEFAULT_WEAPON
    age: float = 0.0
    max_lifetime: float = 1.0
    active: bool = False

    def reset(self, origin, velocity, weapon_type, max_lifetime):
        self.position = [float(origin[0]), float(origin[1]), float(origin[2])]
        self.velocity = [float(velocity[0]), float(velocity[1]), float(velocity[2])]
        self.weapon_type = weapon_type
        self.age = 0.0
        self.max_lifetime = max_lifetime
        self.active = True


def _normalize(vec) -> List[float]:
    mag = math.sqrt(vec[0] ** 2 + vec[1] ** 2 + vec[2] ** 2)
    if mag <= 0.0:
        raise ValueError("方向向量不能为零向量")
    return [vec[0] / mag, vec[1] / mag, vec[2] / mag]


def _magnitude(vec) -> float:
    return math.sqrt(vec[0] ** 2 + vec[1] ** 2 + vec[2] ** 2)


class BulletPhysics:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self._state = {}
        self._history = []
        self.gravity = float(self.config.get("gravity", GRAVITY))
        self._bullets: List[Bullet] = []   # 活跃子弹
        self._pool: List[Bullet] = []      # 对象池（已回收子弹）

    # ------------------------------------------------------------------
    # 弹道计算：重力 + 飞行时间 + 空气阻力（二次阻力模型，欧拉积分）
    # ------------------------------------------------------------------
    def calculate_trajectory(
        self,
        origin,
        direction,
        muzzle_velocity: Optional[float] = None,
        weapon_type: str = DEFAULT_WEAPON,
        dt: float = 0.01,
        max_time: Optional[float] = None,
    ) -> List[Dict]:
        profile = WEAPON_PROFILES.get(weapon_type, WEAPON_PROFILES[DEFAULT_WEAPON])
        v0 = float(muzzle_velocity if muzzle_velocity is not None else profile["muzzle_velocity"])
        if v0 <= 0.0:
            raise ValueError("枪口初速必须为正")
        if dt <= 0.0:
            raise ValueError("积分步长必须为正")
        if max_time is None:
            max_time = profile["max_range"] / v0

        drag_over_mass = profile["drag_coefficient"] / profile["mass"]
        direction = _normalize(direction)

        position = [float(origin[0]), float(origin[1]), float(origin[2])]
        velocity = [direction[0] * v0, direction[1] * v0, direction[2] * v0]

        points = [{
            "time": 0.0,
            "position": list(position),
            "velocity": list(velocity),
            "speed": _magnitude(velocity),
        }]

        t = 0.0
        while t < max_time - 1e-12:
            step = min(dt, max_time - t)
            speed = _magnitude(velocity)
            # 二次阻力的解析衰减因子，显式欧拉在大速度下会数值不稳定
            drag_factor = 1.0 / (1.0 + drag_over_mass * speed * step)
            for axis in range(3):
                velocity[axis] *= drag_factor
            velocity[1] -= self.gravity * step
            for axis in range(3):
                position[axis] += velocity[axis] * step
            t += step
            points.append({
                "time": t,
                "position": list(position),
                "velocity": list(velocity),
                "speed": _magnitude(velocity),
            })
        return points

    # ------------------------------------------------------------------
    # 子弹下坠：按武器类型（枪口初速）与距离计算
    # ------------------------------------------------------------------
    def bullet_drop(
        self,
        distance: float,
        weapon_type: str = DEFAULT_WEAPON,
        muzzle_velocity: Optional[float] = None,
    ) -> float:
        if distance < 0.0:
            raise ValueError("距离不能为负")
        profile = WEAPON_PROFILES.get(weapon_type, WEAPON_PROFILES[DEFAULT_WEAPON])
        v0 = float(muzzle_velocity if muzzle_velocity is not None else profile["muzzle_velocity"])
        if v0 <= 0.0:
            raise ValueError("枪口初速必须为正")
        flight_time = distance / v0
        return 0.5 * self.gravity * flight_time ** 2

    # ------------------------------------------------------------------
    # 风速影响：分解为顺/逆风（影响射程）与横风（造成侧偏），
    # 子弹速度越低受风影响越大
    # ------------------------------------------------------------------
    def wind_effect(
        self,
        wind,
        bullet_velocity,
        flight_time: float,
        reference_speed: float = 100.0,
    ) -> Dict:
        if flight_time < 0.0:
            raise ValueError("飞行时间不能为负")
        speed = _magnitude(bullet_velocity)
        if speed <= 0.0:
            raise ValueError("子弹速度必须为正")

        direction = [bullet_velocity[0] / speed, bullet_velocity[1] / speed, bullet_velocity[2] / speed]
        wind_vec = [float(wind[0]), float(wind[1]), float(wind[2])]

        # 沿弹道方向分量（正=顺风，负=逆风）
        parallel = sum(wind_vec[i] * direction[i] for i in range(3))
        # 垂直分量（横风）
        perpendicular = [wind_vec[i] - parallel * direction[i] for i in range(3)]

        # 低速子弹受风影响更大
        speed_factor = reference_speed / speed
        drift = [component * flight_time * speed_factor for component in perpendicular]
        range_adjustment = parallel * flight_time * speed_factor

        return {
            "drift": drift,
            "lateral_drift": _magnitude(drift),
            "range_adjustment": range_adjustment,
        }

    # ------------------------------------------------------------------
    # 穿透：考虑材质与厚度，计算穿透后的伤害衰减
    # ------------------------------------------------------------------
    def penetration(
        self,
        material: str,
        thickness: float,
        impact_velocity: float,
        base_damage: float = 100.0,
        bullet_mass: float = 0.01,
    ) -> Dict:
        if material not in MATERIALS:
            raise ValueError(f"未知材质 材质: {material}")
        if thickness < 0.0:
            raise ValueError("厚度不能为负")
        if impact_velocity <= 0.0:
            raise ValueError("命中速度必须为正")

        props = MATERIALS[material]
        energy = 0.5 * bullet_mass * impact_velocity ** 2
        required = props["resistance"] * thickness

        if energy <= required:
            return {
                "penetrated": False,
                "damage": 0.0,
                "energy_remaining": 0.0,
            }

        energy_remaining = (energy - required) * props["retention"]
        damage = base_damage * (energy_remaining / energy)
        return {
            "penetrated": True,
            "damage": damage,
            "energy_remaining": energy_remaining,
        }

    # ------------------------------------------------------------------
    # 子弹生命周期：按速度与最大射程计算，过期回收进对象池
    # ------------------------------------------------------------------
    def compute_lifetime(
        self,
        weapon_type: str = DEFAULT_WEAPON,
        muzzle_velocity: Optional[float] = None,
    ) -> float:
        profile = WEAPON_PROFILES.get(weapon_type, WEAPON_PROFILES[DEFAULT_WEAPON])
        v0 = float(muzzle_velocity if muzzle_velocity is not None else profile["muzzle_velocity"])
        if v0 <= 0.0:
            raise ValueError("枪口初速必须为正")
        return profile["max_range"] / v0

    def spawn_bullet(
        self,
        origin,
        direction,
        weapon_type: str = DEFAULT_WEAPON,
        muzzle_velocity: Optional[float] = None,
    ) -> Bullet:
        profile = WEAPON_PROFILES.get(weapon_type, WEAPON_PROFILES[DEFAULT_WEAPON])
        v0 = float(muzzle_velocity if muzzle_velocity is not None else profile["muzzle_velocity"])
        direction = _normalize(direction)
        velocity = [direction[0] * v0, direction[1] * v0, direction[2] * v0]
        lifetime = self.compute_lifetime(weapon_type, v0)

        if self._pool:
            bullet = self._pool.pop()
            bullet.reset(origin, velocity, weapon_type, lifetime)
        else:
            bullet = Bullet()
            bullet.reset(origin, velocity, weapon_type, lifetime)
        self._bullets.append(bullet)
        return bullet

    def update(self, dt: float):
        if dt < 0.0:
            raise ValueError("时间步长不能为负")
        still_active: List[Bullet] = []
        for bullet in self._bullets:
            bullet.age += dt
            profile = WEAPON_PROFILES.get(bullet.weapon_type, WEAPON_PROFILES[DEFAULT_WEAPON])
            drag_over_mass = profile["drag_coefficient"] / profile["mass"]
            speed = _magnitude(bullet.velocity)
            drag_factor = 1.0 / (1.0 + drag_over_mass * speed * dt)
            for axis in range(3):
                bullet.velocity[axis] *= drag_factor
            bullet.velocity[1] -= self.gravity * dt
            for axis in range(3):
                bullet.position[axis] += bullet.velocity[axis] * dt

            if bullet.age >= bullet.max_lifetime or bullet.position[1] < 0.0:
                # 过期或落地：从场景移除并回收进对象池
                bullet.active = False
                self._pool.append(bullet)
            else:
                still_active.append(bullet)
        self._bullets = still_active

    def reset(self):
        self._state = {}
        self._history = []
        for bullet in self._bullets:
            bullet.active = False
            self._pool.append(bullet)
        self._bullets = []

    @property
    def active_count(self) -> int:
        return len(self._bullets)

    @property
    def pool_size(self) -> int:
        return len(self._pool)
