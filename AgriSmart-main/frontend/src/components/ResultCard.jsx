import React, { useState } from "react";

export default function ResultCard({
  title,
  subtitle,
  main,
  confidence,
  bullets,
  imageUrl,
  children,
  accent = "green"
}) {
  const [copiedMessage, setCopiedMessage] = useState("");

  const accentColors = {
    green: { bg: "bg-green-500", gradient: "from-green-400 to-green-600", button: "bg-green-600 hover:bg-green-700", light: "bg-green-50", border: "border-green-200" },
    blue: { bg: "bg-blue-500", gradient: "from-blue-400 to-blue-600", button: "bg-blue-600 hover:bg-blue-700", light: "bg-blue-50", border: "border-blue-200" },
    yellow: { bg: "bg-yellow-500", gradient: "from-yellow-400 to-yellow-600", button: "bg-yellow-600 hover:bg-yellow-700", light: "bg-yellow-50", border: "border-yellow-200" },
  }[accent];

  const formatText = (text) => {
    if (typeof text === "string") {
      return text.replace(/_/g, " ");
    }
    return text;
  };

  const getTextContent = () => {
    if (typeof main === "string") return formatText(main);
    return subtitle ? formatText(subtitle) : title;
  };

  const handleCopy = async () => {
    const text = getTextContent();
    try {
      await navigator.clipboard.writeText(text);
      setCopiedMessage("Copied!");
      setTimeout(() => setCopiedMessage(""), 2000);
    } catch (err) {
      console.error("Failed to copy:", err);
    }
  };

  const handleShare = async () => {
    const text = getTextContent();
    const shareData = {
      title: title,
      text: `${title}: ${text}${confidence ? ` (Confidence: ${(confidence * 100).toFixed(1)}%)` : ""}`
    };

    try {
      if (navigator.share) {
        await navigator.share(shareData);
      } else {
        await navigator.clipboard.writeText(shareData.text);
        setCopiedMessage("Shared!");
        setTimeout(() => setCopiedMessage(""), 2000);
      }
    } catch (err) {
      if (err.name !== "AbortError") {
        console.error("Error sharing:", err);
      }
    }
  };

  return (
    <div className="mt-10 w-full bg-white rounded-3xl shadow-2xl overflow-hidden">
      {/* Header with gradient */}
      <div className={`bg-gradient-to-r ${accentColors.gradient} p-8 text-white`}>
        <div className="flex items-center gap-4">
          <div className="w-20 h-20 rounded-2xl bg-white/20 flex items-center justify-center text-4xl font-bold">
            {title?.charAt(0) || "R"}
          </div>
          <div>
            <h2 className="text-3xl font-bold">{title}</h2>
            {subtitle && <p className="text-white/85 text-lg mt-1">{formatText(subtitle)}</p>}
          </div>
        </div>
      </div>

      {/* Main Content Area */}
      <div className="p-8 md:p-10">
        
        {/* Image and Main Result Side by Side */}
        <div className="grid md:grid-cols-2 gap-8 items-start mb-8">
          
          {/* Left: Main Result & Details */}
          <div className="flex flex-col justify-start">
            {/* Main Result */}
            {main && (
              <div className="mb-8">
                <p className="text-sm font-semibold text-gray-500 uppercase tracking-widest mb-3">Result</p>
                <h3 className={`text-2xl md:text-3xl font-black text-transparent bg-clip-text bg-gradient-to-r ${accentColors.gradient} break-words`}>
                  {formatText(typeof main === "string" ? main : subtitle || title)}
                </h3>
              </div>
            )}

            {/* Confidence Score Card */}
            {typeof confidence === "number" && (
              <div className={`${accentColors.light} border-2 ${accentColors.border} rounded-2xl p-6 mb-8`}>
                <div className="flex items-end justify-between">
                  <div>
                    <p className="text-sm font-semibold text-gray-600 uppercase tracking-wider mb-2">Confidence</p>
                    <p className="text-5xl font-bold text-gray-900">{(confidence * 100).toFixed(1)}%</p>
                  </div>
                  <div className="w-20 h-20 rounded-full flex items-center justify-center" style={{background: accentColors.bg}}>
                    <span className="text-white text-2xl font-bold">{Math.round(confidence * 100)}</span>
                  </div>
                </div>
                
                {/* Progress Bar */}
                <div className="mt-6 w-full bg-gray-300 rounded-full h-3 overflow-hidden">
                  <div
                    className={`h-3 rounded-full bg-gradient-to-r ${accentColors.gradient} transition-all duration-700`}
                    style={{ width: `${Math.min(100, Math.round(confidence * 100))}%` }}
                  />
                </div>
              </div>
            )}

            {/* Recommendations/Bullets */}
            {bullets && bullets.length > 0 && (
              <div className="mb-6">
                <p className="text-sm font-semibold text-gray-500 uppercase tracking-widest mb-4">Key Points</p>
                <ul className="space-y-3">
                  {bullets.map((b, i) => (
                    <li key={i} className="flex gap-3 text-gray-700 text-sm">
                      <span className={`${accentColors.bg} text-white rounded-full w-6 h-6 flex items-center justify-center flex-shrink-0 text-xs font-bold`}>
                        {i + 1}
                      </span>
                      <span className="pt-0.5">{b}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {children}
          </div>

          {/* Right: Image */}
          {imageUrl && (
            <div className="flex flex-col items-center">
              <div className="w-full aspect-square rounded-3xl overflow-hidden border-4 border-gray-200 shadow-lg">
                <img src={imageUrl} alt="result" className="w-full h-full object-cover" />
              </div>
              <p className="text-xs text-gray-500 mt-4">Uploaded image</p>
            </div>
          )}
        </div>

        {/* Action Buttons */}
        <div className="flex gap-4 mt-8 pt-6 border-t border-gray-200">
          <button
            onClick={handleShare}
            className="flex-1 py-4 px-6 bg-gray-100 hover:bg-gray-200 text-gray-800 font-semibold rounded-xl transition duration-300 flex items-center justify-center gap-2 text-lg"
            title="Share result"
          >
            <span>📤</span> Share
          </button>
          <button
            onClick={handleCopy}
            className={`flex-1 py-4 px-6 ${accentColors.button} text-white font-semibold rounded-xl transition duration-300 flex items-center justify-center gap-2 text-lg`}
            title="Copy to clipboard"
          >
            <span>{copiedMessage ? "✓" : "📋"}</span> {copiedMessage || "Copy"}
          </button>
        </div>
      </div>
    </div>
  );
}
