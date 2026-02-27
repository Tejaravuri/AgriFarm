import { useState } from "react";
import { useTranslation } from "react-i18next";
import api from "../api/axios";
import ResultCard from "../components/ResultCard";

export default function Crop() {
  const { t } = useTranslation();
  const [form, setForm] = useState({
    N: "",
    P: "",
    K: "",
    temperature: "",
    humidity: "",
    ph: "",
    rainfall: "",
  });

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleChange = (e) =>
    setForm({ ...form, [e.target.name]: e.target.value });

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const numericForm = {
        N: parseFloat(form.N),
        P: parseFloat(form.P),
        K: parseFloat(form.K),
        temperature: parseFloat(form.temperature),
        humidity: parseFloat(form.humidity),
        ph: parseFloat(form.ph),
        rainfall: parseFloat(form.rainfall)
      };
      
      const token = localStorage.getItem("token");
      if (!token) {
        setError("Please login first");
        setLoading(false);
        return;
      }

      const res = await api.post("/recommend-crop", numericForm);
      setResult(res.data);
    } catch (err) {
      console.error("Crop request error:", err);
      const status = err.response?.status;
      const serverMsg = err.response?.data || err.message;
      setError(
        status ? `Error ${status}: ${JSON.stringify(serverMsg)}` : String(serverMsg || "Recommendation failed")
      );
    }
    setLoading(false);
  };

  return (
    <div className="min-h-screen flex bg-gray-100">
      <div className="hidden md:block w-1/2">
        <img src="/images/crop1.jpg" alt="Crop" className="h-full w-full object-cover" />
      </div>

      <div className="w-full md:w-1/2 flex items-center justify-center p-10">
        <div className="bg-white p-8 rounded-xl shadow-xl w-full max-w-lg">
          <h2 className="text-2xl font-bold text-green-700 mb-6">
            {t('modules.crop.title')}
          </h2>

          <form onSubmit={handleSubmit} className="space-y-4">
            {Object.keys(form).map((key) => {
              const placeholders = {
                N: t('modules.crop.nitrogen'),
                P: t('modules.crop.phosphorus'),
                K: t('modules.crop.potassium'),
                temperature: t('modules.crop.temperature'),
                humidity: t('modules.crop.humidity'),
                ph: t('modules.crop.ph'),
                rainfall: t('modules.crop.rainfall')
              };
              return (
                <input
                  key={key}
                  name={key}
                  placeholder={placeholders[key]}
                  type="number"
                  step="0.01"
                  onChange={handleChange}
                  required
                  className="w-full border p-3 rounded"
                />
              );
            })}

            <button 
              type="submit"
              disabled={loading}
              className="w-full bg-green-600 text-white py-3 rounded hover:bg-green-700 disabled:bg-gray-400"
            >
              {loading ? t('common.loading') : t('modules.crop.submit')}
            </button>
          </form>

          {error && (
            <p className="mt-4 p-3 bg-red-100 text-red-700 rounded">
              ❌ {error}
            </p>
          )}

          {result && (
            <ResultCard
              title="Crop Recommendation"
              subtitle="Best match for your inputs"
              main={<><span className="text-green-700">{result.recommended_crop}</span></>}
              bullets={result.reasons || []}
              accent="green"
            />
          )}
        </div>
      </div>
    </div>
  );
}
