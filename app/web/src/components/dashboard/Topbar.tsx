import type { LucideIcon } from "lucide-react";
import { BatteryMedium, Clock3, RadioTower, Settings } from "lucide-react";
import { Badge } from "../ui/badge";

const metrics: Array<[string, string, LucideIcon?]> = [
  ["Flight Time", "18:42"],
  ["Distance", "2.45 km"],
  ["Coverage", "68 %"],
  ["Battery", "72 %", BatteryMedium],
  ["Link (Telemetry)", "-67 dBm", RadioTower],
  ["Link (Video)", "-52 dBm", RadioTower],
];

export function Topbar() {
  return (
    <header className="grid grid-cols-[minmax(245px,310px)_minmax(0,1fr)_160px] items-center border-b border-border bg-card">
      <div className="flex h-full items-center gap-2 border-r border-border/70 pl-8">
        <strong className="text-sm text-foreground">Mission:</strong>
        <span className="text-sm text-foreground">Maple Street Scan</span>
        <Badge variant="success">Active</Badge>
      </div>
      <div className="grid h-full grid-cols-6">
        {metrics.map(([label, value, Icon]) => (
          <div className="grid min-w-0 content-center justify-center gap-1 border-r border-border/70" key={label}>
            <span className="text-xs text-muted-foreground">{label}</span>
            <strong className="flex items-center justify-center gap-1 text-[15px] font-medium text-foreground">
              {Icon ? <Icon className="text-green-600" size={18} /> : null}
              {value}
            </strong>
          </div>
        ))}
      </div>
      <div className="flex items-center justify-evenly text-[13px] text-muted-foreground">
        <Clock3 className="hidden" size={16} />
        <span>10:24:15 AM</span>
        <Settings size={21} />
      </div>
    </header>
  );
}
