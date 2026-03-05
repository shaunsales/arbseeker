import { NavLink, Outlet } from "react-router-dom";
import { Database, TrendingUp, FlaskConical, Play, Download } from "lucide-react";

const navItems = [
  { to: "/data", label: "Data", icon: Database },
  { to: "/download", label: "Download", icon: Download },
  { to: "/basis", label: "Basis", icon: TrendingUp },
  { to: "/strategy", label: "Strategy", icon: FlaskConical },
  { to: "/backtest", label: "Backtest", icon: Play },
];

export default function AppLayout() {
  return (
    <div className="flex h-screen flex-col bg-gray-950 text-gray-100">
      {/* Top nav */}
      <header className="flex h-14 flex-shrink-0 items-center border-b border-gray-800 px-6">
        <span className="text-sm font-bold tracking-wide text-gray-200">
          Strategy Lab
        </span>
        <nav className="ml-8 flex gap-1">
          {navItems.map(({ to, label, icon: Icon }) => (
            <NavLink
              key={to}
              to={to}
              className={({ isActive }) =>
                `flex items-center gap-1.5 rounded-md px-3 py-1.5 text-sm font-medium transition ${
                  isActive
                    ? "bg-gray-800 text-white"
                    : "text-gray-400 hover:bg-gray-800/50 hover:text-gray-200"
                }`
              }
            >
              <Icon className="h-4 w-4" />
              {label}
            </NavLink>
          ))}
        </nav>
      </header>

      {/* Page content */}
      <main className="flex-1 overflow-hidden">
        <Outlet />
      </main>
    </div>
  );
}
