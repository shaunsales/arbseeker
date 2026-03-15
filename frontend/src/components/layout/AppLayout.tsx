import { NavLink, Outlet, useLocation } from "react-router-dom";
import {
  Database,
  FlaskConical,
  Play,
  ChevronRight,
  Plug,
} from "lucide-react";
import antlyrLogo from "@/assets/antlyr-logo.png";
import {
  Sidebar,
  SidebarContent,
  SidebarGroup,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarInset,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarMenuSub,
  SidebarMenuSubButton,
  SidebarMenuSubItem,
  SidebarProvider,
  SidebarRail,
} from "@/components/ui/sidebar";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";

// ── Nav structure ──

interface NavItem {
  label: string;
  icon: React.ComponentType<{ className?: string }>;
  to?: string;
  children?: { label: string; to: string }[];
}

const navItems: NavItem[] = [
  {
    label: "Data",
    icon: Database,
    children: [
      { label: "Browser", to: "/data" },
      { label: "Download", to: "/download" },
    ],
  },
  {
    label: "Strategies",
    icon: FlaskConical,
    children: [
      { label: "Single Asset", to: "/strategies/single-asset" },
      { label: "Basis", to: "/strategies/basis" },
      { label: "Multi-Leg", to: "/strategies/multi-leg" },
    ],
  },
  {
    label: "Backtest",
    icon: Play,
    children: [
      { label: "Results", to: "/backtest/results" },
      { label: "Execution", to: "/backtest/execution" },
    ],
  },
  {
    label: "CoinRoutes",
    icon: Plug,
    to: "/coinroutes",
  },
];

// ── Layout ──

export default function AppLayout() {
  return (
    <SidebarProvider>
      <AppSidebar />
      <SidebarInset>
        <Outlet />
      </SidebarInset>
    </SidebarProvider>
  );
}

function AppSidebar() {
  const { pathname } = useLocation();

  return (
    <Sidebar collapsible="icon" className="border-r border-sidebar-border">
      <SidebarHeader className="px-4 py-5">
        <NavLink to="/" className="flex items-center gap-3 group-data-[collapsible=icon]:justify-center">
          <img src={antlyrLogo} alt="Antlyr" className="h-8 w-8 shrink-0" />
          <span className="text-base font-bold tracking-widest uppercase text-sidebar-foreground group-data-[collapsible=icon]:hidden">
            Antlyr
          </span>
        </NavLink>
      </SidebarHeader>

      <SidebarContent>
        <SidebarGroup>
          <SidebarGroupLabel>Navigation</SidebarGroupLabel>
          <SidebarMenu>
            {navItems.map((item) =>
              item.children ? (
                <CollapsibleNavItem
                  key={item.label}
                  item={item}
                  pathname={pathname}
                />
              ) : (
                <SidebarMenuItem key={item.label}>
                  <SidebarMenuButton
                    asChild
                    isActive={pathname === item.to}
                    tooltip={item.label}
                  >
                    <NavLink to={item.to!}>
                      <item.icon />
                      <span>{item.label}</span>
                    </NavLink>
                  </SidebarMenuButton>
                </SidebarMenuItem>
              ),
            )}
          </SidebarMenu>
        </SidebarGroup>
      </SidebarContent>

      <SidebarRail />
    </Sidebar>
  );
}

function CollapsibleNavItem({
  item,
  pathname,
}: {
  item: NavItem;
  pathname: string;
}) {
  const isGroupActive = item.children?.some((c) => pathname.startsWith(c.to)) ?? false;

  return (
    <Collapsible asChild defaultOpen={isGroupActive} className="group/collapsible">
      <SidebarMenuItem>
        <CollapsibleTrigger asChild>
          <SidebarMenuButton tooltip={item.label} isActive={isGroupActive}>
            <item.icon />
            <span>{item.label}</span>
            <ChevronRight className="ml-auto h-4 w-4 transition-transform duration-200 group-data-[state=open]/collapsible:rotate-90" />
          </SidebarMenuButton>
        </CollapsibleTrigger>
        <CollapsibleContent>
          <SidebarMenuSub>
            {item.children!.map((child) => (
              <SidebarMenuSubItem key={child.to}>
                <SidebarMenuSubButton
                  asChild
                  isActive={pathname === child.to}
                >
                  <NavLink to={child.to}>{child.label}</NavLink>
                </SidebarMenuSubButton>
              </SidebarMenuSubItem>
            ))}
          </SidebarMenuSub>
        </CollapsibleContent>
      </SidebarMenuItem>
    </Collapsible>
  );
}
