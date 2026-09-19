import numpy as np
import matplotlib.pyplot as plt
import struct
import os
import wave
from matplotlib.animation import FuncAnimation, PillowWriter

def read_spectrogram(filename):
    with open(filename, 'rb') as f:
        num_frames = struct.unpack('i', f.read(4))[0]
        num_bins = struct.unpack('i', f.read(4))[0]
        data = np.frombuffer(f.read(), dtype=np.float32)
        spectrogram = data.reshape((num_frames, num_bins))
    return spectrogram

def read_audio(filename):
    with open(filename, 'rb') as f:
        length = struct.unpack('i', f.read(4))[0]
        audio_data = np.frombuffer(f.read(), dtype=np.float32)
    return audio_data

def main():
    os.makedirs('data', exist_ok=True)
    
    # 1. Export Audio
    if os.path.exists('data/audio.bin'):
        audio = read_audio('data/audio.bin')
        # Normalize audio to [-1, 1] then 16-bit PCM
        audio = audio / np.max(np.abs(audio))
        audio_int16 = (audio * 32767.0).astype(np.int16)
        with wave.open('data/whistler.wav', 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(10000)
            wav_file.writeframes(audio_int16.tobytes())
        print("Exported audio to data/whistler.wav")

    # 2. Export Static & Animated Spectrogram
    if os.path.exists('data/spectrogram.bin'):
        S = read_spectrogram('data/spectrogram.bin')
        S_db = 10 * np.log10(S.T + 1e-10)
        
        # Static plot
        fig, ax = plt.subplots(figsize=(10, 6))
        cax = ax.imshow(S_db, origin='lower', aspect='auto', cmap='magma',
                        extent=[0, S_db.shape[1], 0, 5000]) # 5000 Hz nyquist
        ax.set_title("Whistler Wave Dispersion (M7: Multipath + Noise)")
        ax.set_xlabel("Time (Frames)")
        ax.set_ylabel("Frequency (Hz)")
        plt.colorbar(cax, format='%+2.0f dB')
        plt.tight_layout()
        plt.savefig('data/waterfall_static.png', dpi=150)
        print("Exported static plot to data/waterfall_static.png")
        
        # Animation (Dynamic Graph)
        print("Generating animation (this may take a few seconds)...")
        # To make it cool, we reveal it column by column
        S_anim = np.full_like(S_db, np.nan)
        cax_anim = ax.imshow(S_anim, origin='lower', aspect='auto', cmap='magma',
                             extent=[0, S_db.shape[1], 0, 5000], vmin=np.min(S_db), vmax=np.max(S_db))
        
        def update(frame):
            # Reveal up to 'frame'
            S_anim[:, :frame] = S_db[:, :frame]
            cax_anim.set_data(S_anim)
            return [cax_anim]
        
        frames_to_render = S_db.shape[1]
        step = max(1, frames_to_render // 50) # render ~50 frames for speed
        anim = FuncAnimation(fig, update, frames=range(1, frames_to_render, step), blit=True)
        anim.save('data/waterfall_animated.gif', writer=PillowWriter(fps=15))
        print("Exported dynamic animation to data/waterfall_animated.gif")

if __name__ == '__main__':
    main()
