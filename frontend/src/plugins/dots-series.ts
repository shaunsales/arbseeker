/**
 * Dots Series — a lightweight-charts custom series plugin that renders
 * small filled circles at each data point. Designed for PSAR-style indicators.
 *
 * Uses canvas directly with proper devicePixelRatio scaling so dots look
 * crisp on retina displays without becoming blobs.
 */
import {
  customSeriesDefaultOptions,
  type CustomSeriesPricePlotValues,
  type CustomSeriesOptions,
  type CustomData,
  type ICustomSeriesPaneView,
  type ICustomSeriesPaneRenderer,
  type PaneRendererCustomData,
  type PriceToCoordinateConverter,
  type WhitespaceData,
  type Time,
} from "lightweight-charts";
import type {
  CanvasRenderingTarget2D,
  BitmapCoordinatesRenderingScope,
} from "fancy-canvas";

// ── Options ──

export interface DotsSeriesOptions extends CustomSeriesOptions {
  dotColor: string;
  /** Radius in CSS pixels (will be scaled for retina automatically) */
  radius: number;
}

const defaultOptions: DotsSeriesOptions = {
  ...customSeriesDefaultOptions,
  dotColor: "#f59e0b",
  radius: 2,
};

// ── Renderer ──

export interface DotsData extends CustomData<Time> {
  value: number;
}

class DotsRenderer implements ICustomSeriesPaneRenderer {
  _data: PaneRendererCustomData<Time, DotsData> | null = null;
  _options: DotsSeriesOptions = defaultOptions;

  draw(
    target: CanvasRenderingTarget2D,
    priceConverter: PriceToCoordinateConverter,
  ): void {
    target.useBitmapCoordinateSpace((scope) =>
      this._drawImpl(scope, priceConverter),
    );
  }

  update(
    data: PaneRendererCustomData<Time, DotsData>,
    options: DotsSeriesOptions,
  ): void {
    this._data = data;
    this._options = options;
  }

  _drawImpl(
    scope: BitmapCoordinatesRenderingScope,
    priceToCoordinate: PriceToCoordinateConverter,
  ): void {
    if (
      !this._data ||
      this._data.bars.length === 0 ||
      !this._data.visibleRange
    )
      return;

    const ctx = scope.context;
    const hRatio = scope.horizontalPixelRatio;
    const vRatio = scope.verticalPixelRatio;
    const r = this._options.radius * hRatio; // scale radius for DPI

    ctx.fillStyle = this._options.dotColor;

    for (
      let i = this._data.visibleRange.from;
      i < this._data.visibleRange.to;
      i++
    ) {
      const bar = this._data.bars[i];
      const y = priceToCoordinate(bar.originalData.value);
      if (y === null) continue;

      ctx.beginPath();
      ctx.arc(
        bar.x * hRatio,
        y * vRatio,
        r,
        0,
        Math.PI * 2,
      );
      ctx.fill();
    }
  }
}

// ── Series definition ──

export class DotsSeries
  implements ICustomSeriesPaneView<Time, DotsData, DotsSeriesOptions>
{
  _renderer = new DotsRenderer();

  priceValueBuilder(plotRow: DotsData): CustomSeriesPricePlotValues {
    return [plotRow.value];
  }

  isWhitespace(
    data: DotsData | WhitespaceData<Time>,
  ): data is WhitespaceData<Time> {
    return (data as Partial<DotsData>).value === undefined;
  }

  renderer(): DotsRenderer {
    return this._renderer;
  }

  update(
    data: PaneRendererCustomData<Time, DotsData>,
    options: DotsSeriesOptions,
  ): void {
    this._renderer.update(data, options);
  }

  defaultOptions(): DotsSeriesOptions {
    return { ...defaultOptions };
  }
}
