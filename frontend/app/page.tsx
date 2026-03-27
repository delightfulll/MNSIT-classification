"use client";

import React, { useState, useRef, useCallback } from "react";

type Status = "idle" | "loading" | "success" | "error";

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<number | null>(null);
  const [status, setStatus] = useState<Status>("idle");
  const [errorMsg, setErrorMsg] = useState<string>("");
  const [isDragging, setIsDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFile = (selected: File) => {
    setFile(selected);
    setPrediction(null);
    setStatus("idle");
    setErrorMsg("");
    const reader = new FileReader();
    reader.onload = (e) => setPreview(e.target?.result as string);
    reader.readAsDataURL(selected);
  };

  const onFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selected = e.target.files?.[0];
    if (selected) handleFile(selected);
  };

  const onDrop = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
    const dropped = e.dataTransfer.files?.[0];
    if (dropped) handleFile(dropped);
  }, []);

  const onDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const onDragLeave = () => setIsDragging(false);

  const classify = async () => {
    if (!file) return;
    setStatus("loading");
    setPrediction(null);
    setErrorMsg("");

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("http://localhost:8000/predict", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const data = await res.json();
        throw new Error(data.detail || "Something went wrong");
      }

      const data = await res.json();
      setPrediction(data.prediction);
      setStatus("success");
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : "Unknown error");
      setStatus("error");
    }
  };

  const reset = () => {
    setFile(null);
    setPreview(null);
    setPrediction(null);
    setStatus("idle");
    setErrorMsg("");
    if (inputRef.current) inputRef.current.value = "";
  };

  return (
    <div className="min-h-screen bg-zinc-50 dark:bg-zinc-950 flex items-center justify-center px-4 py-16">
      <div className="w-full max-w-lg">
        {/* Header */}
        <div className="mb-10 text-center">
          <h1 className="text-4xl font-bold tracking-tight text-zinc-900 dark:text-zinc-50">
            MNIST Digit Classifier
          </h1>
          <p className="mt-3 text-zinc-500 dark:text-zinc-400 text-sm">
            Upload a photo of a handwritten digit (0–9) and the model will predict what it is.
          </p>
        </div>

        {/* Drop zone */}
        <div
          onClick={() => inputRef.current?.click()}
          onDrop={onDrop}
          onDragOver={onDragOver}
          onDragLeave={onDragLeave}
          className={`
            relative cursor-pointer rounded-2xl border-2 border-dashed transition-all duration-200
            flex flex-col items-center justify-center gap-3 p-10 text-center
            ${isDragging
              ? "border-blue-500 bg-blue-50 dark:bg-blue-950/30"
              : preview
              ? "border-zinc-300 dark:border-zinc-700 bg-white dark:bg-zinc-900"
              : "border-zinc-300 dark:border-zinc-700 bg-white dark:bg-zinc-900 hover:border-blue-400 hover:bg-blue-50/30 dark:hover:bg-blue-950/10"
            }
          `}
        >
          <input
            ref={inputRef}
            type="file"
            accept="image/*"
            className="hidden"
            onChange={onFileChange}
          />

          {preview ? (
            <>
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={preview}
                alt="Uploaded digit"
                className="max-h-56 max-w-full rounded-xl object-contain shadow-md"
              />
              <p className="text-xs text-zinc-400 dark:text-zinc-500 mt-1">
                {file?.name} · Click to change
              </p>
            </>
          ) : (
            <>
              <div className="w-14 h-14 rounded-full bg-zinc-100 dark:bg-zinc-800 flex items-center justify-center">
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  className="w-7 h-7 text-zinc-400"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                  strokeWidth={1.5}
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5"
                  />
                </svg>
              </div>
              <div>
                <p className="font-medium text-zinc-700 dark:text-zinc-300">
                  Drop an image here
                </p>
                <p className="text-sm text-zinc-400 dark:text-zinc-500 mt-1">
                  or <span className="text-blue-500 hover:underline">browse files</span>
                </p>
              </div>
              <p className="text-xs text-zinc-400 dark:text-zinc-500">
                PNG, JPG, WEBP supported
              </p>
            </>
          )}
        </div>

        {/* Actions */}
        <div className="mt-5 flex gap-3">
          <button
            onClick={classify}
            disabled={!file || status === "loading"}
            className="flex-1 rounded-xl bg-blue-600 hover:bg-blue-700 disabled:bg-zinc-300 dark:disabled:bg-zinc-700 disabled:cursor-not-allowed text-white font-semibold py-3 px-6 transition-colors duration-150 flex items-center justify-center gap-2"
          >
            {status === "loading" ? (
              <>
                <svg className="animate-spin w-4 h-4" viewBox="0 0 24 24" fill="none">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
                </svg>
                Classifying…
              </>
            ) : (
              "Classify Digit"
            )}
          </button>

          {(file || prediction !== null) && (
            <button
              onClick={reset}
              className="rounded-xl border border-zinc-200 dark:border-zinc-700 bg-white dark:bg-zinc-900 text-zinc-600 dark:text-zinc-400 hover:bg-zinc-100 dark:hover:bg-zinc-800 font-medium py-3 px-5 transition-colors duration-150"
            >
              Reset
            </button>
          )}
        </div>

        {/* Result */}
        {status === "success" && prediction !== null && (
          <div className="mt-6 rounded-2xl bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-700 p-6 text-center shadow-sm">
            <p className="text-sm font-medium text-zinc-500 dark:text-zinc-400 uppercase tracking-widest mb-2">
              Predicted Digit
            </p>
            <p className="text-8xl font-bold text-blue-600 dark:text-blue-400 leading-none">
              {prediction}
            </p>
          </div>
        )}

        {/* Error */}
        {status === "error" && (
          <div className="mt-5 rounded-xl bg-red-50 dark:bg-red-950/30 border border-red-200 dark:border-red-800 px-5 py-4 text-sm text-red-700 dark:text-red-400">
            <span className="font-semibold">Error:</span> {errorMsg}
          </div>
        )}
      </div>
    </div>
  );
}
