import { Maximize2 } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "../ui/card";

export function VideoPanel() {
  return (
    <Card className="min-h-0 min-w-0">
      <CardHeader className="pr-4">
        <CardTitle>Live Video</CardTitle>
        <div className="ml-auto flex items-center gap-2 text-[11px] font-bold uppercase text-green-600"><i className="block size-2 rounded-full bg-green-500" />Live</div>
        <Maximize2 size={16} />
      </CardHeader>
      <CardContent>
        <div className="relative h-[113px] overflow-hidden rounded border border-border bg-slate-700">
          <div className="video-road" />
        </div>
        <div className="mt-3 grid grid-cols-[repeat(4,1fr)_28px] gap-2">
          {[0, 1, 2, 3].map((item) => (
            <div className="thumb h-[30px] overflow-hidden rounded" key={item}><span /></div>
          ))}
          <button className="rounded bg-background text-2xl text-foreground shadow">›</button>
        </div>
      </CardContent>
    </Card>
  );
}
