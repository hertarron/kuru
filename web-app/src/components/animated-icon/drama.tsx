"use client";

import { motion, useAnimation, type Variants } from "motion/react";
import type { HTMLAttributes } from "react";
import {
  forwardRef,
  useCallback,
  useId,
  useImperativeHandle,
  useRef,
} from "react";

import { cn } from "@/lib/utils";

export interface DramaIconHandle {
  startAnimation: () => void;
  stopAnimation: () => void;
}

interface DramaIconProps extends HTMLAttributes<HTMLDivElement> {
  size?: number;
}

// Lucide "drama" geometry. The TOP-RIGHT (tragedy) mask is the one the
// owner perceives as "front/top" and it's the animated one; the BOTTOM-LEFT
// (comedy) mask sits static underneath. The knockout carves the tragedy
// silhouette out of the comedy mask so nothing shows through.
const TRAGEDY_OUTLINE =
  "M22 5c0 9-4 12-6 12s-6-3-6-12c0-2 2-3 6-3s6 1 6 3";
const COMEDY_OUTLINE =
  "M10.1 7.1C9 7.2 7.7 7.7 6 8.6c-3.5 2-4.7 3.9-3.7 5.6 4.5 7.8 9.5 8.4 11.2 7.4.9-.5 1.9-2.1 1.9-4.7";

// Counterclockwise splay for the tragedy mask. The knockout group inside
// the <mask> shares this exact variant so the cutout tracks frame-for-frame.
const SPLAY_VARIANTS: Variants = {
  normal: { x: 0, rotate: 0 },
  animate: {
    x: [0, -2, 0.8, 0],
    rotate: [0, -8, 3, 0],
    transition: {
      duration: 0.75,
      times: [0, 0.35, 0.7, 1],
      ease: "easeInOut",
    },
  },
};

const DramaIcon = forwardRef<DramaIconHandle, DramaIconProps>(
  ({ onMouseEnter, onMouseLeave, className, size = 28, ...props }, ref) => {
    const controls = useAnimation();
    const isControlledRef = useRef(false);
    const knockoutId = useId();

    useImperativeHandle(ref, () => {
      isControlledRef.current = true;

      return {
        startAnimation: () => controls.start("animate"),
        stopAnimation: () => controls.start("normal"),
      };
    });

    const handleMouseEnter = useCallback(
      (e: React.MouseEvent<HTMLDivElement>) => {
        if (isControlledRef.current) {
          onMouseEnter?.(e);
        } else {
          controls.start("animate");
        }
      },
      [controls, onMouseEnter]
    );

    const handleMouseLeave = useCallback(
      (e: React.MouseEvent<HTMLDivElement>) => {
        if (isControlledRef.current) {
          onMouseLeave?.(e);
        } else {
          controls.start("normal");
        }
      },
      [controls, onMouseLeave]
    );

    const blink = {
      normal: { opacity: 1 },
      animate: {
        opacity: [1, 0.15, 1],
        transition: { duration: 0.45, times: [0, 0.4, 1], delay: 0.15, ease: "easeInOut" as const },
      },
    };

    return (
      <div
        className={cn(className)}
        onMouseEnter={handleMouseEnter}
        onMouseLeave={handleMouseLeave}
        {...props}
      >
        <svg
          fill="none"
          height={size}
          stroke="currentColor"
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth="2"
          viewBox="0 0 24 24"
          width={size}
          xmlns="http://www.w3.org/2000/svg"
        >
          {/* Static comedy mask FIRST — the moving tragedy mask is painted
              over it, so document order matches the visual stacking. */}
          <g mask={`url(#${knockoutId})`}>
            <path d={COMEDY_OUTLINE} />
            <path d="M9.1 16.5c.3-1.1 1.4-1.7 2.4-1.4" />
            <path d="M10 11h.01" />
            <path d="M6.5 13.1h.01" />
          </g>

          <defs>
            {/* Knockout follows the tragedy mask via the shared splay variant */}
            <mask id={knockoutId} maskUnits="userSpaceOnUse">
              <rect width="24" height="24" fill="#fff" />
              <motion.g animate={controls} initial="normal" variants={SPLAY_VARIANTS}>
                {/* Fat black tragedy silhouette: fill closes the open outline,
                    stroke widens the cutout ~2 units so no seam opens while
                    the mask moves */}
                <path
                  d={TRAGEDY_OUTLINE}
                  fill="#000"
                  stroke="#000"
                  strokeWidth="4"
                />
              </motion.g>
            </mask>
          </defs>

          {/* Tragedy mask on top — splays counterclockwise on hover */}
          <motion.g animate={controls} initial="normal" variants={SPLAY_VARIANTS}>
            <path d={TRAGEDY_OUTLINE} />
            <motion.path
              animate={controls}
              initial="normal"
              variants={{
                normal: { d: "M17.4 9.9c-.8.8-2 .8-2.8 0", opacity: 1 },
                animate: {
                  opacity: 1,
                  d: [
                    "M17.4 9.9c-.8.8-2 .8-2.8 0",
                    "M17.6 10.6c-.9 1.1-2.3 1.1-3.2 0",
                    "M17.4 9.9c-.8.8-2 .8-2.8 0",
                  ],
                  transition: { duration: 0.6, ease: "easeInOut", delay: 0.15 },
                },
              }}
            />
            <motion.path
              animate={controls}
              initial="normal"
              variants={blink}
              d="M18 6h.01"
            />
            <motion.path
              animate={controls}
              initial="normal"
              variants={{ ...blink, animate: { ...blink.animate, transition: { duration: 0.45, times: [0, 0.4, 1], delay: 0.27, ease: "easeInOut" } } }}
              d="M14 6h.01"
            />
          </motion.g>
        </svg>
      </div>
    );
  }
);

DramaIcon.displayName = "Drama";

export { DramaIcon };
