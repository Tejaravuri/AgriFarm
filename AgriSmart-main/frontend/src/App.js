import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";

import Login from "./pages/Login";
import SignUp from "./pages/SignUp";
import Home from "./pages/Home";
import Crop from "./pages/Crop";
import Fertilizer from "./pages/Fertilizer";
import Disease from "./pages/Disease";
import Pest from "./pages/Pest";
import Yield from "./pages/Yield";
import Help from "./pages/Help";

/* 🔐 PROTECTED ROUTE COMPONENT */
function ProtectedRoute({ children }) {
  const token = localStorage.getItem("token");
  return token ? children : <Navigate to="/login" replace />;
}

function App() {
  return (
    <BrowserRouter>
      <Routes>

        {/* 🌍 PUBLIC ROUTES */}
        <Route path="/" element={<Home />} />
        <Route path="/login" element={<Login />} />
        <Route path="/signup" element={<SignUp />} />
        <Route path="/help" element={<Help />} />

        {/* 🔐 PROTECTED MODULES */}
        <Route
          path="/crop"
          element={
            <ProtectedRoute>
              <Crop />
            </ProtectedRoute>
          }
        />
        <Route
          path="/fertilizer"
          element={
            <ProtectedRoute>
              <Fertilizer />
            </ProtectedRoute>
          }
        />
        <Route
          path="/disease"
          element={
            <ProtectedRoute>
              <Disease />
            </ProtectedRoute>
          }
        />
        <Route
          path="/pest"
          element={
            <ProtectedRoute>
              <Pest />
            </ProtectedRoute>
          }
        />
        <Route
          path="/yield"
          element={
            <ProtectedRoute>
              <Yield />
            </ProtectedRoute>
          }
        />

      </Routes>
    </BrowserRouter>
  );
}

export default App;
