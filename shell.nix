{ pkgs ? import <nixpkgs> { } }:
pkgs.mkShell {
  packages = [ pkgs.ffmpeg (pkgs.python3.withPackages (p: with p; [ numpy scikit-learn pytest ])) ];
}
