import { useEffect, useRef } from "react";
import * as echarts from "echarts/core";
import { BarChart, GaugeChart, HeatmapChart, LineChart, ScatterChart } from "echarts/charts";
import {
  DataZoomComponent,
  GridComponent,
  LegendComponent,
  MarkAreaComponent,
  MarkLineComponent,
  MarkPointComponent,
  TooltipComponent,
  VisualMapComponent,
} from "echarts/components";
import { CanvasRenderer } from "echarts/renderers";

echarts.use([
  BarChart,
  LineChart,
  HeatmapChart,
  ScatterChart,
  GaugeChart,
  GridComponent,
  TooltipComponent,
  LegendComponent,
  MarkLineComponent,
  MarkPointComponent,
  MarkAreaComponent,
  VisualMapComponent,
  DataZoomComponent,
  CanvasRenderer,
]);

/** ECharts wrapper (bundled locally: no CDN, works offline). Re-renders when `option` changes. */
export function Chart({ option, height = 260, label, onClick, onReady }: { option: any; height?: number; label: string; onClick?: (p: any) => void; onReady?: (c: echarts.ECharts) => void }) {
  const ref = useRef<HTMLDivElement>(null);
  const inst = useRef<echarts.ECharts | null>(null);
  useEffect(() => {
    if (!ref.current) return;
    inst.current = echarts.init(ref.current, undefined, { renderer: "canvas" });
    const ro = new ResizeObserver(() => inst.current?.resize());
    ro.observe(ref.current);
    return () => {
      ro.disconnect();
      inst.current?.dispose();
      inst.current = null;
    };
  }, []);
  useEffect(() => {
    const c = inst.current;
    if (!c || !onClick) return;
    c.on("click", onClick);
    return () => void c.off("click", onClick);
  }, [onClick]);
  useEffect(() => {
    inst.current?.setOption({ backgroundColor: "transparent", animationDuration: 400, ...option }, true);
    if (inst.current && onReady) onReady(inst.current);
  }, [option]); // eslint-disable-line react-hooks/exhaustive-deps
  return <div ref={ref} role="img" aria-label={label} style={{ width: "100%", height }} />;
}

/** Axis/text colours that follow the light/dark theme. */
export function axisStyle(muted: string, border: string) {
  return {
    axisLine: { lineStyle: { color: border } },
    axisLabel: { color: muted },
    splitLine: { lineStyle: { color: border, opacity: 0.5 } },
  };
}
