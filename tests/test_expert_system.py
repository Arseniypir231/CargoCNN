
"""Автоматические тесты модуля экспертных правил."""

import unittest
from math import pi

from src.expert_system import CargoDetails, evaluate_cargo


class TestExpertSystem(unittest.TestCase):

    def test_rules_for_all_cargo_types(self):
        for cargo in ("barrel", "box", "sack"):
            with self.subTest(cargo=cargo):
                result = evaluate_cargo(cargo, 0.99)
                self.assertEqual(result.final_class, cargo)
                self.assertGreaterEqual(
                    len(result.recommendations), 2
                )

    def test_high_confidence(self):
        self.assertFalse(
            evaluate_cargo("box", 0.96).manual_review_required
        )

    def test_low_confidence(self):
        result = evaluate_cargo("barrel", 0.65)
        self.assertTrue(result.manual_review_required)
        self.assertIsNone(result.final_class)

    def test_confidence_boundary(self):
        self.assertFalse(
            evaluate_cargo("sack", 0.70).manual_review_required
        )

    def test_manual_confirmation(self):
        result = evaluate_cargo(
            "barrel", 0.45, confirmed_class="box"
        )
        self.assertEqual(result.final_class, "box")

    def test_heavy_cargo(self):
        result = evaluate_cargo(
            "box", 0.95, CargoDetails(mass_kg=25)
        )
        self.assertTrue(
            any(
                "грузоподъёмного" in x
                for x in result.recommendations
            )
        )

    def test_light_cargo(self):
        result = evaluate_cargo(
            "box", 0.95, CargoDetails(mass_kg=10)
        )
        self.assertTrue(
            any(
                "способа перемещения" in x
                for x in result.recommendations
            )
        )

    # Проверка объёма коробки.

    def test_dimensions_and_volume(self):
        result = evaluate_cargo(
            "box",
            0.95,
            CargoDetails(
                length_cm=60,
                width_cm=40,
                height_cm=35,
            ),
        )
        self.assertAlmostEqual(result.volume_m3, 0.084)

    # Проверка цилиндрической формы бочки.

    def test_barrel_cylindrical_volume(self):
        result = evaluate_cargo(
            "barrel",
            0.95,
            CargoDetails(
                length_cm=58,
                width_cm=58,
                height_cm=88,
            ),
        )

        expected_volume = (
            pi * (58 / 2) ** 2 * 88 / 1_000_000
        )

        self.assertAlmostEqual(
            result.volume_m3,
            expected_volume,
            places=6,
        )

        self.assertTrue(
            any(
                "цилиндра" in x
                for x in result.recommendations
            )
        )

    # Проверка приблизительного объёма мешка.

    def test_sack_approximate_volume(self):
        result = evaluate_cargo(
            "sack",
            0.95,
            CargoDetails(
                length_cm=60,
                width_cm=40,
                height_cm=35,
            ),
        )

        self.assertAlmostEqual(
            result.volume_m3,
            0.084,
        )

        self.assertTrue(
            any(
                "габаритный объём" in x
                for x in result.recommendations
            )
        )

    # Проверка отсутствующих габаритов.

    def test_incomplete_dimensions(self):
        result = evaluate_cargo(
            "box",
            0.95,
            CargoDetails(
                length_cm=60,
                width_cm=40,
            ),
        )
        self.assertIsNone(result.volume_m3)

    # Проверка отсутствия расчёта до ручного подтверждения.

    def test_low_confidence_does_not_calculate_volume(self):
        result = evaluate_cargo(
            "barrel",
            0.65,
            CargoDetails(
                length_cm=58,
                width_cm=58,
                height_cm=88,
            ),
        )

        self.assertTrue(result.manual_review_required)
        self.assertIsNone(result.final_class)
        self.assertIsNone(result.volume_m3)

    # Проверка выбора формулы после ручного подтверждения.

    def test_confirmed_class_controls_volume_geometry(self):
        result = evaluate_cargo(
            "barrel",
            0.65,
            CargoDetails(
                length_cm=58,
                width_cm=58,
                height_cm=88,
            ),
            confirmed_class="box",
        )

        self.assertEqual(result.predicted_class, "barrel")
        self.assertEqual(result.final_class, "box")
        self.assertFalse(result.manual_review_required)

        self.assertAlmostEqual(
            result.volume_m3,
            58 * 58 * 88 / 1_000_000,
        )

    def test_additional_properties(self):
        result = evaluate_cargo(
            "box",
            0.95,
            CargoDetails(
                fragile=True,
                moisture_sensitive=True,
                keep_upright=True,
            ),
        )
        self.assertEqual(len(result.recommendations), 5)

    def test_without_additional_characteristics(self):
        result = evaluate_cargo("barrel", 0.90)
        self.assertEqual(len(result.recommendations), 2)

    def test_invalid_cargo_type(self):
        with self.assertRaises(ValueError):
            evaluate_cargo("container", 0.95)

    def test_invalid_confidence(self):
        for val in (-0.1, 1.1):
            with self.subTest(value=val):
                with self.assertRaises(ValueError):
                    evaluate_cargo("box", val)

    def test_invalid_threshold(self):
        for val in (0, -0.1, 1.1):
            with self.subTest(value=val):
                with self.assertRaises(ValueError):
                    evaluate_cargo(
                        "box", 0.95, threshold=val
                    )

    def test_invalid_mass(self):
        for val in (0, -5, "тяжёлый"):
            with self.subTest(value=val):
                with self.assertRaises(ValueError):
                    evaluate_cargo(
                        "box",
                        0.95,
                        CargoDetails(mass_kg=val),
                    )

    def test_invalid_property(self):
        with self.assertRaises(ValueError):
            evaluate_cargo(
                "box",
                0.95,
                CargoDetails(fragile="да"),
            )


if __name__ == "__main__":
    unittest.main()
