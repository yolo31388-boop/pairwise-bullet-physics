"""
弹道与子弹物理系统 - 验收测试
运行方式：python -m pytest tests/test_bullet_physics.py -q
共 15 个测试用例
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ballistics.bullet_physics import BulletPhysics


class TestBulletPhysics:
    def setup_method(self):
        self.system = BulletPhysics()

    def test_case_01(self):
        # 弹道计算考虑重力：水平射击的弹道末端应低于枪口水平线
        traj = self.system.calculate_trajectory((0, 0, 0), (1, 0, 0), weapon_type="rifle", distance=500)
        assert len(traj) > 1
        assert traj[-1]["y"] < 0

    def test_case_02(self):
        # 弹道有飞行时间：命中不是瞬间的，且飞行时间随距离增加
        t_near = self.system.flight_time(100, "rifle")
        t_far = self.system.flight_time(1000, "rifle")
        assert t_near > 0
        assert t_far > t_near

    def test_case_03(self):
        # 弹道考虑空气阻力：子弹速度随飞行衰减
        traj = self.system.calculate_trajectory((0, 0, 0), (1, 0, 0), weapon_type="rifle", distance=500)
        assert traj[-1]["speed"] < traj[0]["speed"]

    def test_case_04(self):
        # 子弹下坠按距离计算：距离越远下坠越大，且任何距离都有下坠
        d1 = self.system.bullet_drop(10, "rifle")
        d2 = self.system.bullet_drop(100, "rifle")
        d3 = self.system.bullet_drop(1000, "rifle")
        assert 0 < d1 < d2 < d3

    def test_case_05(self):
        # 子弹下坠按武器类型计算：手枪（低速）比步枪（高速）下坠更大
        drop_pistol = self.system.bullet_drop(100, "pistol")
        drop_rifle = self.system.bullet_drop(100, "rifle")
        assert drop_pistol > drop_rifle

    def test_case_06(self):
        # 风速影响考虑方向：顺风加速、逆风减速，两者不同
        tail = self.system.wind_effect(10, 0, 800, 1.0)
        head = self.system.wind_effect(10, 180, 800, 1.0)
        assert tail["speed_delta"] > 0
        assert head["speed_delta"] < 0
        assert tail["speed_delta"] != head["speed_delta"]

    def test_case_07(self):
        # 横风产生横向偏移，顺风不产生横向偏移
        cross = self.system.wind_effect(10, 90, 800, 1.0)
        tail = self.system.wind_effect(10, 0, 800, 1.0)
        assert abs(cross["lateral_drift"]) > 0
        assert abs(tail["lateral_drift"]) == 0

    def test_case_08(self):
        # 风速影响考虑子弹速度：低速弹比高速弹受横风影响更大
        slow = self.system.wind_effect(10, 90, 300, 1.0)
        fast = self.system.wind_effect(10, 90, 900, 1.0)
        assert abs(slow["lateral_drift"]) > abs(fast["lateral_drift"])

    def test_case_09(self):
        # 穿透计算伤害衰减：穿透后的伤害应小于原始伤害
        result = self.system.penetration("wood", 0.1, 100, "rifle")
        assert result["can_penetrated"]
        assert 0 < result["damage_after"] < 100

    def test_case_10(self):
        # 穿透考虑材质：木头和钢铁的穿透结果不同
        wood = self.system.penetration("wood", 0.1, 100, "rifle")
        steel = self.system.penetration("steel", 0.1, 100, "rifle")
        assert wood["damage_after"] > steel["damage_after"]

    def test_case_11(self):
        # 穿透考虑厚度：薄墙比厚墙穿透后伤害更高
        thin = self.system.penetration("wood", 0.05, 100, "rifle")
        thick = self.system.penetration("wood", 0.5, 100, "rifle")
        assert thin["damage_after"] > thick["damage_after"]

    def test_case_12(self):
        # 子弹生命周期按距离和速度计算，不是写死的10秒
        life_rifle = self.system.bullet_life("rifle")
        life_pistol = self.system.bullet_life("pistol")
        assert life_rifle != 10
        assert life_pistol != 10
        assert life_rifle != life_pistol
        assert self.system.bullet_life("rifle", max_range=500) != self.system.bullet_life("rifle", max_range=1000)

    def test_case_13(self):
        # 子弹过期后从场景消失
        self.system.spawn_bullet((0, 0, 0), (1, 0, 0), weapon_type="rifle", max_range=100)
        assert len(self.system.active_bullets) == 1
        self.system.update(1.0)  # 步枪100m生命周期约0.11秒，1秒后应过期
        assert len(self.system.active_bullets) == 0

    def test_case_14(self):
        # 对象池：过期子弹被回收，新子弹复用同一对象而不是新建
        b1 = self.system.spawn_bullet((0, 0, 0), (1, 0, 0), weapon_type="rifle", max_range=100)
        self.system.update(1.0)
        assert self.system.pool_size == 1
        b2 = self.system.spawn_bullet((0, 0, 0), (1, 0, 0), weapon_type="pistol")
        assert b2 is b1
        assert self.system.pool_size == 0

    def test_case_15(self):
        # reset 清空场景、对象池与历史
        self.system.spawn_bullet((0, 0, 0), (1, 0, 0), weapon_type="rifle", max_range=100)
        self.system.update(1.0)
        self.system.reset()
        assert len(self.system.active_bullets) == 0
        assert self.system.pool_size == 0
        assert self.system._history == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
