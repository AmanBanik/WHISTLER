import numpy as np
import matplotlib.pyplot as plt
import struct
import os
import wave
from matplotlib.animation import FuncAnimation, PillowWriter

def read_spectrogram(filename):
    with open(filename, 'rb') as f:
        magic = f.read(4)
        if magic != b'SPEC':
            raise ValueError("Invalid spectrogram file")
        version = struct.unpack('<i', f.read(4))[0]
        if version != 1:
            raise ValueError(f"Unsupported file version: {version}")
        fs = struct.unpack('<f', f.read(4))[0]
        num_frames = struct.unpack('<i', f.read(4))[0]
        num_bins = struct.unpack('<i', f.read(4))[0]
        data = np.frombuffer(f.read(), dtype=np.float32) # defaults to native but we will assume it's little-endian floats
        spectrogram = data.reshape((num_frames, num_bins))
    return fs, spectrogram

def read_audio(filename):
    with open(filename, 'rb') as f:
        magic = f.read(4)
        if magic != b'WAVA':
            raise ValueError("Invalid audio file")
        version = struct.unpack('<i', f.read(4))[0]
        if version != 1:
            raise ValueError(f"Unsupported file version: {version}")
        fs = struct.unpack('<f', f.read(4))[0]
        length = struct.unpack('<i', f.read(4))[0]
        audio_data = np.frombuffer(f.read(), dtype=np.float32)
    return fs, audio_data

def main():
    os.makedirs('data', exist_ok=True)
    
    # 1. Export Audio
    if os.path.exists('data/audio.bin'):
        fs, audio = read_audio('data/audio.bin')
        # Normalize audio to [-1, 1] then 16-bit PCM (handle zero-amplitude edge case)
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val
        audio_int16 = (audio * 32767.0).astype(np.int16)
        with wave.open('data/whistler.wav', 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(int(fs))
            wav_file.writeframes(audio_int16.tobytes())
        print("Exported audio to data/whistler.wav")

    # 2. Export Static & Animated Spectrogram
    if os.path.exists('data/spectrogram.bin'):
        fs, S = read_spectrogram('data/spectrogram.bin')
        S_db = 10 * np.log10(S.T + 1e-10)
        nyquist = fs / 2.0
        
        # Static plot
        fig_static, ax_static = plt.subplots(figsize=(10, 6))
        cax_static = ax_static.imshow(S_db, origin='lower', aspect='auto', cmap='magma',
                                      extent=[0, S_db.shape[1], 0, nyquist])
        ax_static.set_title("Whistler Wave Dispersion (M7: Multipath + Noise)")
        ax_static.set_xlabel("Time (Frames)")
        ax_static.set_ylabel("Frequency (Hz)")
        plt.colorbar(cax_static, format='%+2.0f dB')
        plt.tight_layout()
        fig_static.savefig('data/waterfall_static.png', dpi=150)
        print("Exported static plot to data/waterfall_static.png")
        plt.close(fig_static)
        
        # Animation (Dynamic Graph)
        print("Generating animation (this may take a few seconds)...")
        fig_anim, ax_anim = plt.subplots(figsize=(10, 6))
        ax_anim.set_title("Whistler Wave Dispersion")
        ax_anim.set_xlabel("Time (Frames)")
        ax_anim.set_ylabel("Frequency (Hz)")
        
        S_anim = np.full_like(S_db, np.nan)
        cax_anim = ax_anim.imshow(S_anim, origin='lower', aspect='auto', cmap='magma',
                                  extent=[0, S_db.shape[1], 0, nyquist], vmin=np.min(S_db), vmax=np.max(S_db))
        plt.colorbar(cax_anim, format='%+2.0f dB')
        plt.tight_layout()
        
        def update(frame):
            S_anim[:, :frame] = S_db[:, :frame]
            cax_anim.set_data(S_anim)
            return [cax_anim]
        
        frames_to_render = S_db.shape[1]
        step = max(1, frames_to_render // 50) 
        anim = FuncAnimation(fig_anim, update, frames=range(1, frames_to_render, step), blit=True)
        anim.save('data/waterfall_animated.gif', writer=PillowWriter(fps=15))
        print("Exported dynamic animation to data/waterfall_animated.gif")
        plt.close(fig_anim)

if __name__ == '__main__':
    main()
