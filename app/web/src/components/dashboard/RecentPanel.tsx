import { inspections } from "../../data/dashboard";
import { Badge } from "../ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "../ui/card";

export function RecentPanel() {
  return (
    <Card className="min-h-0 min-w-0">
      <CardHeader className="pr-4">
        <CardTitle>Recent Inspections</CardTitle>
        <a className="text-[13px] text-primary no-underline">View All</a>
      </CardHeader>
      <CardContent>
        {inspections.map((item) => (
          <div className="flex min-h-[45px] items-center justify-between gap-2.5 border-b border-border" key={item.name}>
            <div>
              <strong className="block text-[13px] font-medium text-foreground">{item.name}</strong>
              <span className="mt-1 block text-xs text-muted-foreground">{item.date} &nbsp; {item.time}</span>
            </div>
            <Badge variant={item.level}>{item.issues}</Badge>
          </div>
        ))}
      </CardContent>
    </Card>
  );
}
