"""
ClassificationPolicy — pet exclusion and target filtering.

Rules (from spec):
  dog  → IGNORED (pet, never alert)
  cat  → IGNORED (pet, never alert)
  configured target animal → ELIGIBLE for alert
  everything else → IGNORED

"Animal" does NOT imply dangerous. Only explicitly configured
target classes are eligible. This class is stateless and pure.
"""
from dataclasses import dataclass
from enum import Enum, auto
from typing import Set


class ClassLabel(Enum):
    PET = auto()       # dog, cat — never alert
    TARGET = auto()    # configured dangerous animal — eligible
    IGNORED = auto()   # everything else — silently skip


@dataclass(frozen=True)
class ClassificationPolicy:
    """
    Classifies a detected class name into PET / TARGET / IGNORED.

    Args:
        target_classes: Set of lower-cased class names that should trigger alerts.
        pet_classes:    Set of lower-cased class names that must never trigger alerts.
    """
    target_classes: frozenset
    pet_classes: frozenset

    @classmethod
    def from_config(
        cls,
        target_classes: list[str],
        pet_classes: list[str] | None = None,
    ) -> "ClassificationPolicy":
        if pet_classes is None:
            pet_classes = ["dog", "cat"]
        return cls(
            target_classes=frozenset(c.lower().strip() for c in target_classes),
            pet_classes=frozenset(c.lower().strip() for c in pet_classes),
        )

    def classify(self, class_name: str) -> ClassLabel:
        """
        Classify a detected class name.

        Args:
            class_name: Raw class string from the model (e.g. 'Wild Boar').

        Returns:
            ClassLabel.PET, ClassLabel.TARGET, or ClassLabel.IGNORED.
        """
        name = class_name.lower().strip()
        if name in self.pet_classes:
            return ClassLabel.PET
        if name in self.target_classes:
            return ClassLabel.TARGET
        return ClassLabel.IGNORED

    def is_target(self, class_name: str) -> bool:
        return self.classify(class_name) == ClassLabel.TARGET

    def is_pet(self, class_name: str) -> bool:
        return self.classify(class_name) == ClassLabel.PET
