import { DamageSummary } from "./components/dashboard/DamageSummary";
import { DetectionsPanel } from "./components/dashboard/DetectionsPanel";
import { InspectionMap } from "./components/dashboard/InspectionMap";
import { MissionProgress } from "./components/dashboard/MissionProgress";
import { RecentPanel } from "./components/dashboard/RecentPanel";
import { Sidebar } from "./components/dashboard/Sidebar";
import { Topbar } from "./components/dashboard/Topbar";
import { VideoPanel } from "./components/dashboard/VideoPanel";

export function App() {
  return (
    <main className="grid h-dvh w-screen grid-cols-[clamp(80px,16vw,250px)_minmax(0,1fr)] overflow-hidden bg-background text-foreground">
      <Sidebar />
      <section className="grid h-dvh min-h-0 min-w-0 grid-rows-[64px_minmax(0,1fr)] overflow-hidden">
        <Topbar />
        <div className="grid h-full min-h-0 grid-cols-[minmax(0,1fr)_clamp(300px,23vw,360px)] gap-3 overflow-hidden p-3">
          <section className="grid min-h-0 min-w-0 grid-rows-[minmax(0,1fr)_minmax(255px,31%)] gap-3 overflow-hidden">
            <InspectionMap />
            <section className="grid min-h-0 min-w-0 grid-cols-[minmax(230px,270px)_minmax(0,1fr)] gap-3 overflow-hidden">
              <MissionProgress />
              <DamageSummary />
            </section>
          </section>
          <aside className="grid min-h-0 min-w-0 grid-rows-[minmax(190px,26%)_minmax(268px,34%)_minmax(178px,1fr)] gap-3 overflow-hidden">
            <VideoPanel />
            <DetectionsPanel />
            <RecentPanel />
          </aside>
        </div>
      </section>
    </main>
  );
}
