import { useState, useEffect, useContext } from "react";
import axios from "axios";
import { AuthContext } from "../context/AuthContext";


export default function YieldForm() {
const [form, setForm] = useState({
location: "",
crop: "",
year: "",
rainfall: "",
temperature: "",
soil_ph: ""
});


const [result, setResult] = useState(null);
const [locations, setLocations] = useState([]);
const [crops, setCrops] = useState([]);

const { user: token } = useContext(AuthContext);


const handleChange = (e) => {
setForm({ ...form, [e.target.name]: e.target.value });
};


const handleSubmit = async (e) => {
	e.preventDefault();
	try {
		const headers = {};
		if (token) headers["Authorization"] = `Bearer ${token}`;
		const res = await axios.post("http://127.0.0.1:5000/predict-yield", form, { headers });
		setResult(res.data);
	} catch (err) {
		console.error(err);
		alert(err.response?.data?.error || "Prediction failed");
	}
};

useEffect(() => {
	async function fetchEncoders() {
		try {
			const res = await axios.get("http://127.0.0.1:5000/encoders");
			setLocations(res.data.locations || []);
			setCrops(res.data.crops || []);
		} catch (e) {
			console.warn("Could not fetch encoders:", e.message || e);
		}
	}
	fetchEncoders();
}, []);

return (
<div className="max-w-xl mx-auto bg-white p-6 rounded-2xl shadow-lg">
<h2 className="text-2xl font-bold text-green-700 mb-4">🌾 Yield Prediction</h2>


<form onSubmit={handleSubmit} className="space-y-3">
<div className="space-y-3">
    <select name="location" value={form.location} onChange={handleChange} className="w-full p-2 border rounded" required>
        <option value="">Select Location (Country)</option>
        {locations.map((loc) => (
            <option key={loc} value={loc}>{loc}</option>
        ))}
    </select>

    <input name="crop" placeholder="Enter Crop (e.g., Wheat, Rice, Maize)" value={form.crop} onChange={handleChange} className="w-full p-2 border rounded" required />

    <div className="grid grid-cols-2 gap-3">
        <input name="year" placeholder="Year" value={form.year} onChange={handleChange} className="p-2 border rounded" required />
        <input name="rainfall" placeholder="Rainfall (mm)" value={form.rainfall} onChange={handleChange} className="p-2 border rounded" required />

        <input name="temperature" placeholder="Temperature (°C)" value={form.temperature} onChange={handleChange} className="p-2 border rounded" required />
        <input name="soil_ph" placeholder="Soil pH" value={form.soil_ph} onChange={handleChange} className="p-2 border rounded" required />
    </div>
</div>


<button className="w-full bg-green-600 text-white p-2 rounded-lg hover:bg-green-700">
Predict Yield
</button>
</form>


{result && (
<div className="mt-5 bg-green-50 p-4 rounded-lg">
<p className="font-semibold">🌱 Expected Yield:</p>
<p className="text-xl text-green-800">
{result.predicted_yield_tons_per_hectare} tons/hectare
</p>


<p className="font-semibold mt-3">📌 Care Advisory:</p>
<ul className="list-disc ml-5">
{result.care_advisory.map((item, i) => (
<li key={i}>{item}</li>
))}
</ul>
</div>
)}
</div>
);
}