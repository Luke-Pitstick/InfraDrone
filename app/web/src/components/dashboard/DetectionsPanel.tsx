import { AlertTriangle } from "lucide-react";
import { detections } from "../../data/dashboard";
import { Card, CardContent, CardHeader, CardTitle } from "../ui/card";

export function DetectionsPanel() {
  return (
    <Card className="min-h-0 min-w-0">
      <CardHeader className="pr-4">
        <CardTitle>Detections <span>(Today)</span></CardTitle>
        <a className="text-[13px] text-primary no-underline">12</a>
      </CardHeader>
      <CardContent>
        {detections.map((item) => (
          <div className="grid min-h-[47px] grid-cols-[25px_1fr_97px_36px] items-center gap-1.5 border-b border-border text-xs" key={item.issue}>
            <AlertTriangle className={item.severity === "High" ? "text-red-500" : item.severity === "Medium" ? "text-yellow-500" : "text-green-600"} size={20} />
            <div>
              <strong className={item.severity === "High" ? "block text-[13px] text-red-600" : item.severity === "Medium" ? "block text-[13px] text-amber-600" : "block text-[13px] text-green-600"}>{item.severity}</strong>
              <span className="block text-foreground">{item.issue}</span>
            </div>
            <em className="block text-muted-foreground not-italic">{item.location}</em>
            <b className={item.severity === "High" ? "grid h-6 min-w-[34px] place-items-center justify-self-end rounded border border-red-200 bg-red-50 text-red-700" : item.severity === "Medium" ? "grid h-6 min-w-[34px] place-items-center justify-self-end rounded border border-amber-200 bg-amber-50 text-amber-700" : "grid h-6 min-w-[34px] place-items-center justify-self-end rounded border border-green-200 bg-green-50 text-green-700"}>{item.score}</b>
          </div>
        ))}
        <button className="flex h-[34px] w-full items-center justify-between bg-transparent text-primary">View All Detections <span>→</span></button>
      </CardContent>
    </Card>
  );
}
