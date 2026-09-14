from __future__ import annotations


class PlanningObjective:
    def compute(self, outputs, ctx):
        raise NotImplementedError
