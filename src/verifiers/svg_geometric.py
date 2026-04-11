"""SVG Geometric Verifier — Phase 1: 4 composable predicates.

Pipeline: SVG string → svgelements DOM+transform → Shapely geometry → Diagnostic list.

Predicates:
  1. bbox_coverage: 根元素 viewBox 中被子元素覆盖的比例
  2. element_overlap: 任意元素对之间的重叠面积比
  3. spatial_containment: 子元素是否完全在父容器内
  4. size_proportion: 元素尺寸相对于 viewBox 的比例是否合理
"""

from __future__ import annotations

import io
import itertools
import logging
from dataclasses import dataclass, field
from typing import Any

from shapely.geometry import box as shapely_box
from shapely.ops import unary_union
from shapely.validation import make_valid
from svgelements import SVG, Circle, Ellipse, Group, Rect

from .diagnostic import Diagnostic, Severity

logger = logging.getLogger(__name__)


@dataclass
class PredicateConfig:
    """单个 predicate 的阈值配置。"""
    bbox_coverage_min: float = 0.1       # viewBox 最低覆盖率
    overlap_max: float = 0.3             # 最大允许重叠比 (IoU)
    containment_strict: bool = True      # 是否要求严格包含
    size_ratio_range: tuple[float, float] = (0.01, 0.95)  # 元素/viewBox 面积比范围


@dataclass
class ParsedElement:
    """解析后的 SVG 元素。"""
    elem_id: str
    geometry: Any           # Shapely geometry
    parent_id: str | None
    area: float
    bbox: tuple[float, float, float, float]  # (minx, miny, maxx, maxy)
    tag: str                # 元素类型 (rect, circle, path, ...)
    is_group: bool = False
    children: list[str] = field(default_factory=list)


class SVGGeometricVerifier:
    """SVG 几何验证器，输出 Diagnostic 列表。

    Usage:
        verifier = SVGGeometricVerifier(config=PredicateConfig())
        diagnostics = verifier.verify(svg_string)
    """

    def __init__(self, config: PredicateConfig | None = None):
        self.config = config or PredicateConfig()

    def verify(self, svg_string: str) -> list[Diagnostic]:
        """对 SVG 字符串执行全部 4 个 predicate 检查。"""
        parsed = self._parse_svg(svg_string)
        if parsed is None:
            return [Diagnostic(
                rule_id="parse_error",
                severity=Severity.ERROR,
                message_generic="SVG parsing failed",
                message_precise="Could not parse SVG: invalid or empty SVG content",
            )]

        viewbox_geom, elements = parsed
        if not elements:
            return []

        diagnostics: list[Diagnostic] = []
        diagnostics.extend(self._check_bbox_coverage(viewbox_geom, elements))
        diagnostics.extend(self._check_element_overlap(elements))
        diagnostics.extend(self._check_spatial_containment(elements))
        diagnostics.extend(self._check_size_proportion(viewbox_geom, elements))
        return diagnostics

    def _parse_svg(
        self, svg_string: str
    ) -> tuple[Any, list[ParsedElement]] | None:
        """解析 SVG 为 viewbox geometry + 元素列表。

        Returns:
            (viewbox_shapely_geom, [ParsedElement, ...]) or None if parse fails.
        """
        try:
            svg = SVG.parse(io.StringIO(svg_string))
        except Exception as e:
            logger.warning("SVG parse failed: %s", e)
            return None

        # Extract viewBox
        if not hasattr(svg, "viewbox"):
            logger.warning("Parsed SVG has no viewbox attribute (type=%s)", type(svg).__name__)
            return None
        vb = svg.viewbox
        if vb is None or vb.width <= 0 or vb.height <= 0:
            # Fallback to width/height attributes
            w = getattr(svg, "width", 0) or 0
            h = getattr(svg, "height", 0) or 0
            if w <= 0 or h <= 0:
                logger.warning("No valid viewBox or width/height")
                return None
            viewbox_geom = shapely_box(0, 0, float(w), float(h))
        else:
            viewbox_geom = shapely_box(
                float(vb.x), float(vb.y),
                float(vb.x + vb.width), float(vb.y + vb.height),
            )

        elements: list[ParsedElement] = []
        elem_counter = 0
        parent_stack: list[str] = []

        for elem in svg.elements():
            if elem is svg:
                continue

            # Determine element id
            raw_id = getattr(elem, "id", None) or ""
            if not raw_id:
                elem_counter += 1
                raw_id = f"_auto_{elem_counter}"

            # Track parent via group nesting
            is_group = isinstance(elem, Group)
            parent_id = parent_stack[-1] if parent_stack else None

            if is_group:
                parent_stack.append(raw_id)
                elements.append(ParsedElement(
                    elem_id=raw_id,
                    geometry=None,
                    parent_id=parent_id,
                    area=0,
                    bbox=(0, 0, 0, 0),
                    tag="group",
                    is_group=True,
                ))
                continue

            # Extract bounding box via svgelements (handles transforms)
            geom = self._elem_to_shapely(elem)
            if geom is None or geom.is_empty:
                continue

            geom = make_valid(geom)
            b = geom.bounds  # (minx, miny, maxx, maxy)

            tag = type(elem).__name__.lower()
            elements.append(ParsedElement(
                elem_id=raw_id,
                geometry=geom,
                parent_id=parent_id,
                area=geom.area,
                bbox=b,
                tag=tag,
            ))

        # Filter to only leaf elements with geometry
        leaf_elements = [e for e in elements if not e.is_group and e.geometry is not None]
        return viewbox_geom, leaf_elements

    @staticmethod
    def _elem_to_shapely(elem: Any) -> Any | None:
        """将 svgelements 元素转换为 Shapely geometry。

        Uses bounding box as universal fallback; exact geometry for
        Rect, Circle, Ellipse.
        """
        try:
            if isinstance(elem, Rect):
                x = float(elem.x)
                y = float(elem.y)
                w = float(elem.width)
                h = float(elem.height)
                if w > 0 and h > 0:
                    return shapely_box(x, y, x + w, y + h)

            if isinstance(elem, Circle):
                cx = float(elem.cx)
                cy = float(elem.cy)
                r = float(elem.r)
                if r > 0:
                    from shapely.geometry import Point
                    return Point(cx, cy).buffer(r, resolution=16)

            if isinstance(elem, Ellipse):
                cx = float(elem.cx)
                cy = float(elem.cy)
                rx = float(elem.rx)
                ry = float(elem.ry)
                if rx > 0 and ry > 0:
                    from shapely.geometry import Point
                    from shapely import affinity
                    circle = Point(0, 0).buffer(1.0, resolution=16)
                    return affinity.scale(
                        affinity.translate(circle, cx, cy), rx, ry
                    )

            # Fallback: use svgelements bbox
            bbox_result = elem.bbox()
            if bbox_result is not None:
                xmin, ymin, xmax, ymax = bbox_result
                if xmax > xmin and ymax > ymin:
                    return shapely_box(float(xmin), float(ymin),
                                       float(xmax), float(ymax))
        except Exception as e:
            logger.debug("Failed to convert element to Shapely: %s", e)
        return None

    def _check_bbox_coverage(
        self, viewbox_geom: Any, elements: list[ParsedElement]
    ) -> list[Diagnostic]:
        """Predicate 1: 子元素对 viewBox 的覆盖率。"""
        vb_area = viewbox_geom.area
        if vb_area <= 0:
            return []

        geoms = [e.geometry for e in elements if e.geometry is not None]
        if not geoms:
            return []

        try:
            union = unary_union(geoms)
            covered = viewbox_geom.intersection(union).area
        except Exception:
            return []

        coverage = covered / vb_area

        if coverage < self.config.bbox_coverage_min:
            return [Diagnostic(
                rule_id="bbox_coverage",
                severity=Severity.WARNING,
                message_generic="Low viewBox coverage detected",
                message_precise=(
                    f"Elements cover only {coverage:.1%} of the viewBox "
                    f"(threshold: {self.config.bbox_coverage_min:.0%}). "
                    "Consider redistributing elements or adjusting viewBox dimensions."
                ),
                metric_name="coverage_pct",
                metric_value=round(coverage * 100, 2),
                fix_direction=(
                    "Increase element sizes, redistribute elements to fill "
                    "viewBox, or shrink viewBox to match content"
                ),
            )]
        return []

    def _check_element_overlap(
        self, elements: list[ParsedElement]
    ) -> list[Diagnostic]:
        """Predicate 2: 元素对之间的重叠检测 (IoU)。"""
        diagnostics: list[Diagnostic] = []
        leaf = [e for e in elements if e.area > 0]

        # Group siblings by parent_id
        siblings: dict[str | None, list[ParsedElement]] = {}
        for e in leaf:
            siblings.setdefault(e.parent_id, []).append(e)

        for group_elements in siblings.values():
            for a, b in itertools.combinations(group_elements, 2):
                try:
                    inter = a.geometry.intersection(b.geometry).area
                except Exception:
                    continue
                if inter <= 0:
                    continue
                union_area = a.area + b.area - inter
                if union_area <= 0:
                    continue
                iou = inter / union_area

                if iou > self.config.overlap_max:
                    diagnostics.append(Diagnostic(
                        rule_id="element_overlap",
                        severity=Severity.WARNING,
                        element_ids=(a.elem_id, b.elem_id),
                        message_generic="Element overlap detected",
                        message_precise=(
                            f"Elements '{a.elem_id}' and '{b.elem_id}' overlap "
                            f"with IoU={iou:.1%} (threshold: {self.config.overlap_max:.0%}). "
                            f"Intersection area: {inter:.1f}."
                        ),
                        metric_name="overlap_iou",
                        metric_value=round(iou, 4),
                        fix_direction=(
                            f"Move '{a.elem_id}' or '{b.elem_id}' apart, "
                            "or reduce their sizes to decrease overlap"
                        ),
                        metadata={"intersection_area": round(inter, 2)},
                    ))
        return diagnostics

    def _check_spatial_containment(
        self, elements: list[ParsedElement]
    ) -> list[Diagnostic]:
        """Predicate 3: 子元素是否在父容器内。"""
        diagnostics: list[Diagnostic] = []
        elem_map = {e.elem_id: e for e in elements}

        for e in elements:
            if e.parent_id is None or e.parent_id not in elem_map:
                continue
            parent = elem_map[e.parent_id]
            if parent.geometry is None or e.geometry is None:
                continue

            try:
                if e.geometry.within(parent.geometry):
                    continue
                # Calculate overflow
                overflow = e.geometry.difference(parent.geometry)
                if overflow.is_empty:
                    continue
                overflow_pct = overflow.area / e.area if e.area > 0 else 0
            except Exception:
                continue

            if not self.config.containment_strict and overflow_pct < 0.05:
                continue

            # Determine overflow direction from centroids
            ec = overflow.centroid
            pc = parent.geometry.centroid
            dx = ec.x - pc.x
            dy = ec.y - pc.y
            direction = []
            if abs(dx) > abs(dy):
                direction.append("right" if dx > 0 else "left")
            else:
                direction.append("bottom" if dy > 0 else "top")
            dir_str = direction[0]

            diagnostics.append(Diagnostic(
                rule_id="spatial_containment",
                severity=Severity.ERROR,
                element_ids=(e.elem_id, e.parent_id),
                message_generic="Element extends beyond its container",
                message_precise=(
                    f"Element '{e.elem_id}' extends {overflow_pct:.1%} beyond "
                    f"container '{e.parent_id}' on the {dir_str} side."
                ),
                metric_name="overflow_pct",
                metric_value=round(overflow_pct * 100, 2),
                fix_direction=(
                    f"Move '{e.elem_id}' toward the center of '{e.parent_id}' "
                    f"or reduce its size to fit within the container"
                ),
                metadata={"overflow_direction": dir_str,
                          "overflow_area": round(overflow.area, 2)},
            ))
        return diagnostics

    def _check_size_proportion(
        self, viewbox_geom: Any, elements: list[ParsedElement]
    ) -> list[Diagnostic]:
        """Predicate 4: 元素尺寸相对 viewBox 是否合理。"""
        diagnostics: list[Diagnostic] = []
        vb_area = viewbox_geom.area
        if vb_area <= 0:
            return []

        lo, hi = self.config.size_ratio_range

        for e in elements:
            if e.area <= 0:
                continue
            ratio = e.area / vb_area

            if ratio < lo:
                diagnostics.append(Diagnostic(
                    rule_id="size_proportion",
                    severity=Severity.INFO,
                    element_ids=(e.elem_id,),
                    message_generic="Element is disproportionately small",
                    message_precise=(
                        f"Element '{e.elem_id}' occupies {ratio:.2%} of viewBox area "
                        f"(minimum threshold: {lo:.0%}). May be invisible or a rendering artifact."
                    ),
                    metric_name="size_ratio",
                    metric_value=round(ratio * 100, 4),
                    fix_direction=(
                        f"Scale up '{e.elem_id}' or remove it if unintentional"
                    ),
                ))
            elif ratio > hi:
                diagnostics.append(Diagnostic(
                    rule_id="size_proportion",
                    severity=Severity.WARNING,
                    element_ids=(e.elem_id,),
                    message_generic="Element is disproportionately large",
                    message_precise=(
                        f"Element '{e.elem_id}' occupies {ratio:.2%} of viewBox area "
                        f"(maximum threshold: {hi:.0%}). May obscure other elements."
                    ),
                    metric_name="size_ratio",
                    metric_value=round(ratio * 100, 4),
                    fix_direction=(
                        f"Scale down '{e.elem_id}' to leave room for other elements"
                    ),
                ))
        return diagnostics
