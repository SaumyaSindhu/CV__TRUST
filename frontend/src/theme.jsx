import React, { createContext, useContext, useEffect, useMemo, useState } from "react";

/* Recharts needs literal colour values (it writes them into SVG
 * attributes), so the CSS tokens in index.css are mirrored here. Keep the
 * two in sync when changing the palette. */
const DARK = {
  bg: "#0a0a0b",
  panel: "#121214",
  panel2: "#1a1a1d",
  line: "#27272a",
  grid: "#232326",
  ink: "#d4d4d8",
  muted: "#71717a",
  head: "#fafafa",
  ok: "#22c55e",
  warn: "#f59e0b",
  bad: "#ef4444",
  info: "#60a5fa",
  accent: "#f97316",
  accent2: "#fdba74",
  violet: "#a78bfa",
  tooltipBg: "#18181b",
};

const LIGHT = {
  bg: "#e9e9ec",
  panel: "#ffffff",
  panel2: "#f4f4f5",
  line: "#e4e4e7",
  grid: "#e9e9ec",
  ink: "#3f3f46",
  muted: "#71717a",
  head: "#09090b",
  ok: "#16a34a",
  warn: "#d97706",
  bad: "#dc2626",
  info: "#2563eb",
  accent: "#ea580c",
  accent2: "#9a3412",
  violet: "#7c3aed",
  tooltipBg: "#ffffff",
};

const STORAGE_KEY = "cv-assurance-theme";
const ThemeContext = createContext({ dark: true, setDark: () => {}, C: DARK });

export function ThemeProvider({ children }) {
  const [dark, setDark] = useState(() => {
    try {
      return window.localStorage.getItem(STORAGE_KEY) !== "light";
    } catch {
      return true;
    }
  });

  useEffect(() => {
    const root = document.documentElement;
    root.classList.toggle("dark", dark);
    root.classList.toggle("light", !dark);
    try {
      window.localStorage.setItem(STORAGE_KEY, dark ? "dark" : "light");
    } catch {
      /* storage unavailable (private mode / air-gapped kiosk) — ignore */
    }
  }, [dark]);

  const value = useMemo(() => ({ dark, setDark, C: dark ? DARK : LIGHT }), [dark]);
  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>;
}

export const useTheme = () => useContext(ThemeContext);

/** Shared recharts tooltip styling that follows the active theme. */
export function tooltipProps(C) {
  return {
    contentStyle: {
      background: C.tooltipBg,
      border: `1px solid ${C.line}`,
      borderRadius: 12,
      fontSize: 12,
      color: C.ink,
      boxShadow: "0 10px 30px rgba(0,0,0,0.35)",
    },
    labelStyle: { color: C.muted, marginBottom: 4 },
    itemStyle: { color: C.ink },
    cursor: { fill: "rgba(249,115,22,0.08)" },
  };
}
