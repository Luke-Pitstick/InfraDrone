import L from "leaflet";
import { Circle, MapContainer, Marker, Polyline, TileLayer, useMap } from "react-leaflet";
import { Crosshair, Layers, Minus, Plus } from "lucide-react";
import { Button } from "../ui/button";
import { mapDetections, scanCenter, scanTrack } from "../../data/dashboard";

const droneIcon = L.divIcon({
  className: "leaflet-drone-icon",
  html: `<div class="leaflet-drone-pulse"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 2 20 22 12 18 4 22 12 2Z" /></svg><span></span></div>`,
  iconSize: [116, 116],
  iconAnchor: [58, 58],
});

function detectionIcon(tone: string, label: string) {
  return L.divIcon({
    className: `leaflet-detection-icon ${tone}`,
    html: `<div aria-label="${label} detection"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 9v4M12 17h.01M10.3 3.9 1.8 18.3A2 2 0 0 0 3.5 21h17a2 2 0 0 0 1.7-2.7L13.7 3.9a2 2 0 0 0-3.4 0Z" /></svg></div>`,
    iconSize: [30, 38],
    iconAnchor: [15, 36],
  });
}

export function InspectionMap() {
  return (
    <section className="grid min-h-0 min-w-0 grid-rows-[45px_1fr] overflow-hidden">
      <div className="flex h-[45px] items-end border-b border-border">
        <button className="h-[45px] border-b-2 border-primary px-6 text-xs font-bold uppercase text-primary">Live Map</button>
        <button className="h-[45px] border-b-2 border-transparent px-6 text-xs font-bold uppercase text-muted-foreground">Orthomosaic</button>
        <button className="h-[45px] border-b-2 border-transparent px-6 text-xs font-bold uppercase text-muted-foreground">3D View</button>
      </div>
      <div className="relative min-h-0 min-w-0 overflow-hidden bg-slate-900">
        <MapContainer center={scanCenter} zoom={18} zoomControl={false} attributionControl={false} className="leaflet-map">
          <TileLayer url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}" />
          <Polyline positions={scanTrack} pathOptions={{ color: "#f3c229", weight: 5, opacity: 0.9 }} />
          <Polyline positions={scanTrack.map(([lat, lng]) => [lat, lng + 0.000035] as [number, number])} pathOptions={{ color: "#2fb86f", weight: 2, opacity: 0.9 }} />
          <Circle center={scanCenter} radius={34} pathOptions={{ color: "#1f8cff", fillColor: "#1f8cff", fillOpacity: 0.12, weight: 1 }} />
          <Marker position={scanCenter} icon={droneIcon} />
          {mapDetections.map((item) => (
            <Marker key={`${item.position[0]}-${item.position[1]}`} position={item.position} icon={detectionIcon(item.tone, item.label)} />
          ))}
          <MapActionControls />
        </MapContainer>
        <button className="absolute right-[18px] top-[18px] z-[480] h-8 rounded-md border border-border bg-background/95 px-3 text-foreground shadow-lg">Orthomosaic (Latest)</button>
        <div className="scale">10 m</div>
        <div className="absolute bottom-11 right-3.5 z-[480] grid w-[134px] gap-2.5 rounded border border-border bg-background/95 px-3.5 py-3 text-[13px] text-foreground shadow-lg">
          <strong className="text-[11px] uppercase tracking-[0.04em]">Damage Severity</strong>
          <span className="flex items-center gap-2"><i className="block size-[13px] rounded-full bg-red-500" />High</span>
          <span className="flex items-center gap-2"><i className="block size-[13px] rounded-full bg-orange-500" />Medium</span>
          <span className="flex items-center gap-2"><i className="block size-[13px] rounded-full bg-yellow-400" />Low</span>
          <span className="flex items-center gap-2"><i className="block size-[13px] rounded-full bg-green-500" />Low / Monitor</span>
        </div>
      </div>
    </section>
  );
}

function MapActionControls() {
  const map = useMap();

  return (
    <div className="absolute left-[18px] top-[18px] z-[480] grid gap-2.5">
      <Button size="icon" variant="ghost" aria-label="Zoom in" onClick={() => map.zoomIn()}>
        <Plus />
      </Button>
      <Button size="icon" variant="ghost" aria-label="Zoom out" onClick={() => map.zoomOut()}>
        <Minus />
      </Button>
      <Button size="icon" variant="ghost" aria-label="Center map" onClick={() => map.setView(scanCenter, 18)}>
        <Crosshair />
      </Button>
      <Button size="icon" variant="ghost" aria-label="Layers">
        <Layers />
      </Button>
    </div>
  );
}
