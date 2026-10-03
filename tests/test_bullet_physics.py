"""
弹道与子弹物理系统 - 验收测试
运行方式：python -m pytest tests/test_bullet_physics.py -q
共 15 个测试用例
"""
import math
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ballistics.bullet_physics import BulletPhysics, GRAVITY


class TestBulletPhysics:
    def setup_method(self):
        self.system = BulletPhysics()

    # 1. 弹道必须有飞行时间：子弹不是瞬间命中
    def test_case_01(self):
        traj = self.system.calculate_trajectory(
            origin=(0, 0, 0), direction=(1, 0, 0), max_time=1.0
        )
        assert len(traj) > 2
        assert traj[0]["time"] == 0.0
        times = [p["time"] for p in traj]
        assert all(b > a for a, b in zip(times, times[1:]))
        assert traj[-1]["time"] == pytest.approx(1.0)

    # 2. 弹道必须受重力影响：水平射击时高度不断下降
    def test_case_02(self):
        traj = self.system.calculate_trajectory(
            origin=(0, 100, 0), direction=(1, 0, 0), max_time=1.0
        )
        assert traj[-1]["position"][1] < traj[0]["position"][1]
        drop = traj[0]["position"][1] - traj[-1]["position"][1]
        assert drop > 1.0  # 1秒自由落体约4.9米，必须明显下坠

    # 3. 弹道必须受空气阻力影响：速度随时间衰减
    def test_case_03(self):
        traj = self.system.calculate_trajectory(
            origin=(0, 100, 0), direction=(1, 0, 0), max_time=1.0
        )
        assert traj[-1]["speed"] < traj[0]["speed"]

    # 4. 下坠必须随距离增加：1000米下坠远大于10米
    def test_case_04(self):
        drop_near = self.system.bullet_drop(10.0)
        drop_far = self.system.bullet_drop(1000.0)
        assert drop_near > 0.0
        assert drop_far > drop_near * 100  # 下坠与距离平方成正比

    # 5. 下坠必须考虑武器类型：手枪初速低，下坠大于步枪
    def test_case_05(self):
        drop_pistol = self.system.bullet_drop(100.0, weapon_type="pistol")
        drop_rifle = self.system.bullet_drop(100.0, weapon_type="rifle")
        assert drop_pistol > drop_rifle > 0.0

    # 6. 横风必须造成侧向偏移
    def test_case_06(self):
        result = self.system.wind_effect(
            wind=(0, 0, 5.0), bullet_velocity=(800, 0, 0), flight_time=1.0
        )
        assert result["lateral_drift"] > 0.0
        assert abs(result["drift"][2]) > 0.0

    # 7. 顺风与逆风必须不同：顺风增程、逆风减程
    def test_case_07(self):
        tailwind = self.system.wind_effect(
            wind=(10, 0, 0), bullet_velocity=(800, 0, 0), flight_time=1.0
        )
        headwind = self.system.wind_effect(
            wind=(-10, 0, 0), bullet_velocity=(800, 0, 0), flight_time=1.0
        )
        assert tailwind["range_adjustment"] > 0.0
        assert headwind["range_adjustment"] < 0.0
        assert tailwind["range_adjustment"] == pytest.approx(
            -headwind["range_adjustment"]
        )

    # 8. 风速影响必须考虑子弹速度：低速子弹偏移更大
    def test_case_08(self):
        slow = self.system.wind_effect(
            wind=(0, 0, 5.0), bullet_velocity=(300, 0, 0), flight_time=1.0
        )
        fast = self.system.wind_effect(
            wind=(0, 0, 5.0), bullet_velocity=(900, 0, 0), flight_time=1.0
        )
        assert slow["lateral_drift"] > fast["lateral_drift"] > 0.0

    # 9. 穿透必须计算伤害衰减：穿透后伤害低于基础伤害
    def test_case_09(self):
        result = self.system.penetration(
            material="wood", thickness=0.05, impact_velocity=800.0, base_damage=100.0
        )
        assert result["penetrated"] is True
        assert 0.0 < result["damage"] < 100.0

    # 10. 穿透必须考虑材质：钢铁比木头难穿透、伤害衰减更大
    def test_case_10(self):
        wood = self.system.penetration(
            material="wood", thickness=0.05, impact_velocity=800.0
        )
        steel = self.system.penetration(
            material="steel", thickness=0.05, impact_velocity=800.0
        )
        assert wood["penetrated"] is True
        assert steel["penetrated"] is False
        assert wood["damage"] > steel["damage"]

    # 11. 穿透必须考虑厚度：厚墙伤害衰减更大，过厚则无法穿透
    def test_case_11(self):
        thin = self.system.penetration(
            material="wood", thickness=0.02, impact_velocity=800.0
        )
        thick = self.system.penetration(
            material="wood", thickness=0.10, impact_velocity=800.0
        )
        very_thick = self.system.penetration(
            material="wood", thickness=5.0, impact_velocity=800.0
        )
        assert thin["damage"] > thick["damage"] > 0.0
        assert very_thick["penetrated"] is False
        assert very_thick["damage"] == 0.0

    # 12. 子弹生命周期必须按速度与射程计算，不能写死
    def test_case_12(self):
        life_rifle = self.system.compute_lifetime(weapon_type="rifle")
        life_sniper = self.system.compute_lifetime(weapon_type="sniper")
        life_pistol = self.system.compute_lifetime(weapon_type="pistol")
        assert life_rifle != life_sniper != life_pistol
        assert life_rifle != 10.0  # 不再是写死的10秒
        assert life_sniper > life_pistol  # 狙击枪射程远，生命周期更长

    # 13. 过期子弹必须从场景中消失
    def test_case_13(self):
        bullet = self.system.spawn_bullet(
            origin=(0, 100, 0), direction=(1, 0, 0), weapon_type="pistol"
        )
        assert self.system.active_count == 1
        self.system.update(bullet.max_lifetime + 0.1)
        assert self.system.active_count == 0
        assert bullet.active is False

    # 14. 必须有对象池：过期子弹被回收并复用，不频繁新建对象
    def test_case_14(self):
        bullet = self.system.spawn_bullet(
            origin=(0, 100, 0), direction=(1, 0, 0), weapon_type="pistol"
        )
        self.system.update(bullet.max_lifetime + 0.1)
        assert self.system.pool_size == 1
        reused = self.system.spawn_bullet(
            origin=(0, 100, 0), direction=(1, 0, 0), weapon_type="rifle"
        )
        assert reused is bullet  # 复用同一对象，没有新建
        assert self.system.pool_size == 0
        assert reused.active is True
        assert reused.age == 0.0

    # 15. update 必须推进子弹物理：位置前进且受重力下坠
    def test_case_15(self):
        bullet = self.system.spawn_bullet(
            origin=(0, 100, 0), direction=(1, 0, 0), weapon_type="rifle"
        )
        x0, y0 = bullet.position[0], bullet.position[1]
        self.system.update(0.1)
        assert bullet.position[0] > x0  # 子弹在飞行
        assert bullet.position[1] < y0  # 受重力下坠
        assert bullet.age == pytest.approx(0.1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
