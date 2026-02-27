import { Link } from "react-router-dom";

export default function ModuleCard({ title, description, image, link }) {
  return (
    <Link to={link}>
      <div className="bg-white rounded-2xl shadow-lg overflow-hidden hover:scale-105 transition-transform duration-300">
        
        <img
          src={image}
          alt={title}
          className="h-40 w-full object-cover"
        />

        <div className="p-4">
          <h3 className="text-xl font-bold text-green-700">{title}</h3>
          <p className="text-gray-600 mt-2 text-sm">{description}</p>
        </div>

      </div>
    </Link>
  );
}
