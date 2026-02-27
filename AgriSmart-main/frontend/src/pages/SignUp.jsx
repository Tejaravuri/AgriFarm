import React, { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useTranslation } from "react-i18next";
import api from "../api/axios";

export default function SignUp() {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const [form, setForm] = useState({ name: "", email: "", password: "" });

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      await api.post("/signup", form);
      alert("Account created successfully! Please login.");
      navigate("/login");
    } catch (err) {
      console.error("Signup error:", err);
      const errorMsg = err.response?.data?.error || "Signup failed. Please try again.";
      alert(errorMsg);
    }
  };

  return (
    <div
      className="min-h-screen flex items-center justify-center bg-cover bg-center"
      style={{ backgroundImage: "url('/images/login.jpg')" }}
    >
      <div className="bg-white/80 backdrop-blur-lg shadow-2xl rounded-2xl p-10 w-[380px]">
        <h2 className="text-3xl font-bold text-center text-green-700 mb-6">
          {t('auth.signupTitle')}
        </h2>

        <form onSubmit={handleSubmit}>
          <input
            placeholder={t('auth.namePlaceholder')}
            className="w-full mb-4 px-4 py-3 rounded-lg border focus:outline-none focus:ring-2 focus:ring-green-600"
            onChange={(e) => setForm({ ...form, name: e.target.value })}
            required
          />

          <input
            type="email"
            placeholder={t('auth.emailPlaceholder')}
            className="w-full mb-4 px-4 py-3 rounded-lg border focus:outline-none focus:ring-2 focus:ring-green-600"
            onChange={(e) => setForm({ ...form, email: e.target.value })}
            required
          />

          <input
            type="password"
            placeholder={t('auth.passwordPlaceholder')}
            className="w-full mb-6 px-4 py-3 rounded-lg border focus:outline-none focus:ring-2 focus:ring-green-600"
            onChange={(e) => setForm({ ...form, password: e.target.value })}
            required
          />

          <button type="submit" className="w-full bg-green-700 hover:bg-green-800 text-white py-3 rounded-lg font-semibold transition">
            {t('auth.signupButton')}
          </button>
        </form>

        <p className="text-center text-sm mt-6">
          {t('auth.haveAccount')}{" "}
          <Link to="/" className="text-green-700 font-semibold">
            {t('auth.loginLink')}
          </Link>
        </p>
      </div>
    </div>
  );
}
