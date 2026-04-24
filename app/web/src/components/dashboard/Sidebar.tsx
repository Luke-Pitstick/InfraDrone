import type { LucideIcon } from "lucide-react";
import {
  Compass,
  Crosshair,
  FileText,
  Gauge,
  History,
  LocateFixed,
  Map,
  MenuSquare,
  Navigation,
  Settings,
  Signal,
  Sun,
  UploadCloud,
  Wind,
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "../ui/card";

const navItems: Array<{ label: string; icon: LucideIcon; active?: boolean }> = [
  { label: "Dashboard", icon: MenuSquare, active: true },
  { label: "Map", icon: Map },
  { label: "Missions", icon: LocateFixed },
  { label: "Detections", icon: Crosshair },
  { label: "Reports", icon: FileText },
  { label: "History", icon: History },
  { label: "Settings", icon: Settings },
];

const statusRows: Array<[string, string, LucideIcon, "green" | ""]> = [
  ["Armed", "GUIDED", Signal, "green"],
  ["GPS", "3D Fix (12)", LocateFixed, "green"],
  ["Altitude", "24.6 m AGL", Gauge, ""],
  ["Speed", "8.7 m/s", Gauge, ""],
  ["Heading", "178°", Compass, ""],
  ["Home Distance", "412 m", Crosshair, ""],
];

export function Sidebar() {
  return (
    <aside className="flex min-h-0 flex-col gap-3 overflow-hidden border-r border-border bg-card px-3 py-4">
      <div className="mb-5 grid grid-cols-[42px_1fr] items-center gap-2 px-1">
        <div className="grid place-items-center text-primary">
          <Navigation size={29} strokeWidth={1.9} />
        </div>
        <div>
          <h1 className="m-0 text-2xl font-bold leading-none tracking-normal text-foreground">RoadEye</h1>
          <p className="mt-1 text-[11px] text-muted-foreground">Autonomous Road Inspection</p>
        </div>
      </div>
      <nav className="grid gap-2" aria-label="Primary">
        {navItems.map((item) => (
          <button
            key={item.label}
            className={
              item.active
                ? "flex h-10 items-center gap-4 rounded-md bg-primary px-3 text-white"
                : "flex h-10 items-center gap-4 rounded-md px-3 text-slate-600 hover:bg-accent"
            }
          >
            <item.icon size={21} />
            <span>{item.label}</span>
          </button>
        ))}
      </nav>
      <VehicleStatus />
      <WeatherCard />
    </aside>
  );
}

function VehicleStatus() {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Vehicle Status</CardTitle>
      </CardHeader>
      <CardContent className="grid gap-3">
        {statusRows.map(([label, value, Icon, tone]) => (
          <div className="grid grid-cols-[21px_1fr_auto] items-center text-[13px] text-muted-foreground" key={label}>
            <span className={tone === "green" ? "flex text-green-600" : "flex text-muted-foreground"}>
              <Icon size={16} />
            </span>
            <strong className={tone === "green" ? "font-semibold text-green-600" : "font-semibold text-foreground"}>{label}</strong>
            <span>{value}</span>
          </div>
        ))}
      </CardContent>
    </Card>
  );
}

function WeatherCard() {
  return (
    <Card className="mt-auto">
      <CardHeader>
        <CardTitle>Weather</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid h-[54px] grid-cols-[36px_1fr_auto] items-center gap-2.5 border-b border-border">
          <Sun className="text-yellow-400" size={31} />
          <strong className="text-xl text-foreground">22 °C</strong>
          <span className="text-[13px] text-muted-foreground">Sunny</span>
        </div>
        <div className="grid h-9 grid-cols-[28px_1fr_auto] items-center text-[13px] text-muted-foreground">
          <Wind size={18} />
          <span>Wind</span>
          <strong className="font-medium text-muted-foreground">6 m/s NE</strong>
        </div>
        <div className="grid h-9 grid-cols-[28px_1fr_auto] items-center text-[13px] text-muted-foreground">
          <UploadCloud size={17} />
          <span>Humidity</span>
          <strong className="font-medium text-muted-foreground">42%</strong>
        </div>
      </CardContent>
    </Card>
  );
}
