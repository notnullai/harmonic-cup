import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "@fontsource-variable/inter";
import "../theme.css";
import "./try.css";
import App from "./App";
import BenchmarkPage from "./BenchmarkPage";

// One entry (web/try/index.html, api/main.py's `/try/` and `/try/benchmark`
// routes both serve it), routed client-side by pathname: the wizard has its
// own run-id-in-fragment routing already (App.tsx), and the benchmark page
// needs no state, so a plain pathname check is all this needs.
const isBenchmark = window.location.pathname.replace(/\/+$/, "") === "/try/benchmark";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    {isBenchmark ? <BenchmarkPage /> : <App />}
  </StrictMode>,
);
