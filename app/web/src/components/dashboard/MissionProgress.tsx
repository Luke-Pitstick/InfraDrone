import { Button } from "../ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "../ui/card";

export function MissionProgress() {
  return (
    <Card className="min-h-0 min-w-0">
      <CardHeader>
        <CardTitle>Mission Progress</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="my-3 grid grid-cols-[1fr_46px] items-center gap-3">
          <span className="h-2 overflow-hidden rounded-full bg-slate-200"><i className="block h-full w-[68%] rounded-full bg-green-500" /></span>
          <strong className="text-foreground">68%</strong>
        </div>
        <div className="mb-3 flex justify-between text-[13px] text-muted-foreground"><span>Area Scanned</span><strong className="font-medium text-muted-foreground">2.45 km / 3.60 km</strong></div>
        <div className="mb-3 flex justify-between text-[13px] text-muted-foreground"><span>Est. Time Remaining</span><strong className="font-medium text-muted-foreground">14:32</strong></div>
        <div className="mb-3 flex justify-between text-[13px] text-muted-foreground"><span>Next Waypoint</span><strong className="font-medium text-muted-foreground">WP-17</strong></div>
        <div className="mt-3 flex gap-3">
          <Button className="min-w-0 flex-1 text-[13px]" variant="outline">Pause Mission</Button>
          <Button className="min-w-0 flex-1 text-[13px]" variant="destructive">End Mission</Button>
        </div>
      </CardContent>
    </Card>
  );
}
