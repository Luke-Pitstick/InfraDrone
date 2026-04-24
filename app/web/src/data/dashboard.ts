export type Detection = {
  severity: "High" | "Medium" | "Low";
  issue: string;
  location: string;
  score: number;
};

export type Inspection = {
  name: string;
  date: string;
  time: string;
  issues: string;
  level: "danger" | "warning" | "success";
};

export type MapDetection = {
  position: [number, number];
  tone: "red" | "orange" | "green";
  label: string;
};

export const detections: Detection[] = [
  { severity: "High", issue: "Alligator Cracking", location: "Maple St - Seg 42", score: 92 },
  { severity: "High", issue: "Pothole", location: "Maple St - Seg 37", score: 88 },
  { severity: "Medium", issue: "Longitudinal Crack", location: "Maple St - Seg 33", score: 63 },
  { severity: "Low", issue: "Transverse Crack", location: "Maple St - Seg 28", score: 35 },
];

export const inspections: Inspection[] = [
  { name: "Maple Street Scan", date: "May 26, 2025", time: "9:12 AM", issues: "12 Issues", level: "danger" },
  { name: "Oak Avenue Scan", date: "May 24, 2025", time: "2:45 PM", issues: "7 Issues", level: "warning" },
  { name: "Pine Road Scan", date: "May 22, 2025", time: "11:08 AM", issues: "3 Issues", level: "warning" },
  { name: "Cedar Lane Scan", date: "May 20, 2025", time: "10:30 AM", issues: "1 Issue", level: "success" },
];

export const scanCenter: [number, number] = [40.01493, -105.27067];

export const scanTrack: [number, number][] = [
  [40.01398, -105.27105],
  [40.01434, -105.27091],
  [40.01472, -105.27078],
  [40.01513, -105.27063],
  [40.01552, -105.27049],
  [40.01587, -105.27036],
];

export const mapDetections: MapDetection[] = [
  { position: [40.01567, -105.27041], tone: "red", label: "High" },
  { position: [40.01464, -105.27097], tone: "green", label: "Low" },
  { position: [40.0143, -105.27098], tone: "red", label: "High" },
  { position: [40.01448, -105.2706], tone: "orange", label: "Medium" },
];
