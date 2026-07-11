import { useState } from "react";
import { hasToken, setToken } from "./api";
import Dashboard from "./Dashboard";
import Login from "./Login";

export default function App() {
  const [name, setName] = useState<string | null>(
    hasToken() ? sessionStorage.getItem("evolve_name") : null,
  );

  if (!name) {
    return (
      <Login
        onLogin={(n) => {
          sessionStorage.setItem("evolve_name", n);
          setName(n);
        }}
      />
    );
  }
  return (
    <Dashboard
      displayName={name}
      onLogout={() => {
        setToken(null);
        sessionStorage.removeItem("evolve_name");
        setName(null);
      }}
    />
  );
}
