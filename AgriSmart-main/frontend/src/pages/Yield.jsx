import { useState } from "react";
import { useTranslation } from "react-i18next";
import api from "../api/axios";
import ResultCard from "../components/ResultCard";

export default function Yield() {
  const { t } = useTranslation();
  const locations = ['Afghanistan', 'Albania', 'Algeria', 'Argentina', 'Australia', 'Brazil', 'Canada', 'China', 'Colombia', 'Egypt', 'Finland', 'France', 'Germany', 'Ghana', 'India', 'Indonesia', 'Italy', 'Japan', 'Kenya', 'Mexico', 'Netherlands', 'Nigeria', 'Pakistan', 'Peru', 'Philippines', 'Poland', 'Russia', 'South Africa', 'Spain', 'Thailand', 'Turkey', 'Ukraine', 'United Kingdom', 'United States', 'Vietnam', 'Zimbabwe'];
  
  const crops = ['Cassava', 'Maize', 'Plantains and others', 'Potatoes', 'Rice, paddy', 'Sorghum', 'Soybeans', 'Sweet potatoes', 'Wheat', 'Yams'];
  
  const [form, setForm] = useState({
    location: "",
    crop: "",
    year: new Date().getFullYear(),
    rainfall: "",
    temperature: "",
    soil_ph: "",
  });

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleChange = (e) => {
    setForm({ ...form, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    const token = localStorage.getItem("token");
    if (!token) {
      setError("Please login first");
      return;
    }

    setLoading(true);
    setError(null);
    try {
      const dataToSend = {
        location: form.location,
        crop: form.crop,
        year: parseInt(form.year),
        rainfall: parseFloat(form.rainfall),
        temperature: parseFloat(form.temperature),
        soil_ph: parseFloat(form.soil_ph || 6.5)
      };
      
      const res = await api.post("/predict-yield", dataToSend);
      setResult(res.data);
    } catch (err) {
      console.error("Yield request error:", err);
      const status = err.response?.status;
      const serverMsg = err.response?.data || err.message;
      setError(
        status ? `Error ${status}: ${JSON.stringify(serverMsg)}` : String(serverMsg || "Prediction failed. Please check inputs.")
      );
    }
    setLoading(false);
  };

  return (
    <div className="min-h-screen flex bg-gray-100">

      {/* 🌾 LEFT IMAGE PANEL */}
      <div className="hidden md:flex w-[45%] bg-green-700 items-center justify-center">
        <img
          src="/images/crop.jpg"
          alt="Yield Prediction"
          className="h-full w-full object-cover"
        />
      </div>

      {/* 🧠 RIGHT FORM PANEL */}
      <div className="w-full md:w-[70%] flex items-center justify-center p-8">
        <div className="w-full max-w-3xl bg-white rounded-2xl shadow-xl p-10">

          <h1 className="text-3xl font-bold text-green-700 mb-2">
            {t('modules.yield.title')}
          </h1>
          <p className="text-gray-600 mb-8">
            {t('modules.yield.subtitle')}
          </p>

          <form onSubmit={handleSubmit} className="space-y-6">

            <div className="grid md:grid-cols-2 gap-5">
              <select
                name="location"
                value={form.location}
                onChange={handleChange}
                required
                className="border p-3 rounded-lg"
              >
                <option value="">{t('modules.yield.selectLocation')}</option>
                {locations.map((loc) => (
                  <option key={loc} value={loc}>
                    {loc}
                  </option>
                ))}
              </select>

              <select
                name="crop"
                value={form.crop}
                onChange={handleChange}
                required
                className="border p-3 rounded-lg"
              >
                <option value="">{t('modules.yield.selectCrop')}</option>
                {crops.map((c) => (
                  <option key={c} value={c}>
                    {c}
                  </option>
                ))}
              </select>

              <input
                name="year"
                placeholder={t('modules.yield.year')}
                onChange={handleChange}
                required
                className="border p-3 rounded-lg"
              />

              <input
                name="rainfall"
                placeholder={t('modules.crop.rainfall')}
                onChange={handleChange}
                required
                className="border p-3 rounded-lg"
              />

              <input
                name="temperature"
                placeholder={t('modules.crop.temperature')}
                onChange={handleChange}
                required
                className="border p-3 rounded-lg"
              />

              <input
                name="soil_ph"
                placeholder={t('modules.crop.ph')}
                onChange={handleChange}
                className="border p-3 rounded-lg"
              />
            </div>

            <button
              type="submit"
              disabled={loading}
              className="w-full bg-green-600 text-white py-3 rounded-lg text-lg font-semibold hover:bg-green-700 transition disabled:bg-gray-400"
            >
              {loading ? t('common.loading') : t('modules.yield.submit')}
            </button>
          </form>

          {error && (
            <div className="mt-4 p-3 bg-red-100 text-red-700 rounded">
              ❌ {error}
            </div>
          )}

          {/* 📊 RESULT */}
          {result && (
            <ResultCard
              title="Yield Prediction"
              subtitle="Smart crop forecast"
              main={<span className="text-green-700 font-bold">{result.predicted_yield_tons_per_hectare} tons/hectare</span>}
              bullets={result.care_advisory && Array.isArray(result.care_advisory) ? result.care_advisory : 
                       (result.care_advisory && typeof result.care_advisory === "string" ? [result.care_advisory] : [])}
              accent="green"
            />
          )}
        </div>
      </div>
    </div>
  );
}
