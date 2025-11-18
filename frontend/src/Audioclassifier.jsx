import {React, useState } from "react";
import { Upload, Music, Loader } from "lucide-react";
import "./index.css";

const API_URL = "https://anisssh--audio-cnn-fastapi-app.modal.run/predict";

export default function AudioClassifier() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [dragOver, setDragOver] = useState(false);

  const handleFile = (selectedFile) => {
    if (selectedFile && selectedFile.type.startsWith("audio/")) {
      setFile(selectedFile);
      setResult(null);
      setError(null);
    } else {
      setError("Please select a valid audio file");
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    const droppedFile = e.dataTransfer.files[0];
    handleFile(droppedFile);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setDragOver(true);
  };

  const handleDragLeave = () => {
    setDragOver(false);
  };

  const classifyAudio = async () => {
    if (!file) return;

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const formData = new FormData();
      formData.append("audio", file);

      const response = await fetch(API_URL, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(
        err.message + ". Make sure to update API_URL with your Modal endpoint!"
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-linear-to-br from-purple-600 to-indigo-700 flex items-center justify-center p-4">
      <div className="bg-white rounded-3xl shadow-2xl p-8 w-full max-w-lg">
        <div className="flex items-center gap-3 mb-2">
          <Music className="w-8 h-8 text-purple-600" />
          <h1 className="text-3xl font-bold text-gray-800">Audio Classifier</h1>
        </div>
        <p className="text-gray-600 mb-6 text-sm">
          Upload an audio file to classify environmental sounds
        </p>

        {/* Upload Area */}
        <div
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onClick={() => document.getElementById("fileInput").click()}
          className={`border-3 border-dashed rounded-2xl p-12 text-center cursor-pointer transition-all ${
            dragOver
              ? "border-indigo-600 bg-indigo-50 scale-105"
              : "border-purple-400 bg-purple-50 hover:bg-purple-100"
          }`}
        >
          <Upload className="w-12 h-12 text-purple-600 mx-auto mb-4" />
          <p className="text-purple-700 font-semibold mb-1">
            Click to upload or drag and drop
          </p>
          <p className="text-gray-500 text-sm">WAV, MP3, OGG files supported</p>
          <input
            id="fileInput"
            type="file"
            accept="audio/*"
            onChange={(e) => handleFile(e.target.files[0])}
            className="hidden"
          />
        </div>

        {/* File Info */}
        {file && (
          <div className="mt-6 p-4 bg-purple-50 rounded-xl">
            <p className="text-gray-700 font-medium mb-3">📄 {file.name}</p>
            <button
              onClick={classifyAudio}
              disabled={loading}
              className="w-full bg-linear-to-r from-purple-600 to-indigo-600 text-white py-3 rounded-xl font-semibold hover:shadow-lg transition-all disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? "Classifying..." : "Classify Audio"}
            </button>
          </div>
        )}

        {/* Loading Spinner */}
        {loading && (
          <div className="flex justify-center mt-6">
            <Loader className="w-10 h-10 text-purple-600 animate-spin" />
          </div>
        )}

        {/* Error */}
        {error && (
          <div className="mt-6 p-4 bg-red-50 border border-red-200 rounded-xl">
            <p className="text-red-700 text-sm">{error}</p>
          </div>
        )}

        {/* Result */}
        {result && (
          <div className="mt-6 p-6 bg-linear-to-br from-purple-50 to-indigo-50 rounded-2xl animate-fadeIn">
            <p className="text-xs text-gray-500 uppercase tracking-wider mb-2">
              Prediction
            </p>
            <p className="text-3xl font-bold text-purple-700 mb-4 capitalize">
              {result.prediction}
            </p>

            <p className="text-xs text-gray-500 uppercase tracking-wider mb-2">
              Confidence
            </p>
            <p className="text-xl font-semibold text-gray-800 mb-2">
              {(result.confidence * 100).toFixed(1)}%
            </p>

            <div className="w-full h-2 bg-gray-200 rounded-full overflow-hidden">
              <div
                className="h-full bg-linear-to-r from-purple-600 to-indigo-600 rounded-full transition-all duration-500"
                style={{ width: `${result.confidence * 100}%` }}
              />
            </div>
          </div>
        )}
      </div>

      <style>{`
        @keyframes fadeIn {
          from {
            opacity: 0;
            transform: translateY(10px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
        .animate-fadeIn {
          animation: fadeIn 0.5s ease-out;
        }
      `}</style>
    </div>
  );
}
