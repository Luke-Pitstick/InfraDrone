import { Card, CardContent, CardHeader, CardTitle } from "../ui/card";

export function DamageSummary() {
  return (
    <Card className="min-h-0 min-w-0">
      <CardHeader>
        <CardTitle>Damage Summary <span>(Today)</span></CardTitle>
      </CardHeader>
      <CardContent>
        <div className="mb-3 grid min-w-0 grid-cols-4 gap-3">
          <StatBox label="High Severity" value="4" pct="33%" tone="red" />
          <StatBox label="Medium Severity" value="5" pct="42%" tone="orange" />
          <StatBox label="Low Severity" value="3" pct="25%" tone="green" />
          <StatBox label="Total Detections" value="12" pct="100%" tone="blue" />
        </div>
        <div className="chart" aria-label="Detection trend chart">
          {Array.from({ length: 22 }).map((_, i) => (
            <span key={i} style={{ height: `${10 + ((i * 7) % 36) + i * 1.7}px` }} />
          ))}
          <div className="chart-fill" />
        </div>
        <div className="axis"><span>09:00</span><span>09:30</span><span>10:00</span><span>10:30</span><span>11:00</span></div>
      </CardContent>
    </Card>
  );
}

function StatBox({ label, value, pct, tone }: { label: string; value: string; pct: string; tone: string }) {
  const toneClass = {
    red: "border-red-200 bg-red-50 text-red-700",
    orange: "border-amber-200 bg-amber-50 text-amber-700",
    green: "border-green-200 bg-green-50 text-green-700",
    blue: "border-border bg-slate-50 text-foreground",
  }[tone];

  return (
    <div className={`grid min-h-[65px] grid-cols-[1fr_auto] items-end rounded border p-3 ${toneClass}`}>
      <span className="col-span-2 text-xs font-bold">{label}</span>
      <strong className="text-[29px] leading-none text-current">{value}</strong>
      <em className="text-xs not-italic text-muted-foreground">{pct}</em>
    </div>
  );
}
