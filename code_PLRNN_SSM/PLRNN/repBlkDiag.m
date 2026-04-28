function [ BigM ] = repBlkDiag( M, number_rep )
% repeats the Matrix M NUMBER_REP times in the block diagonal

MCell = repmat({M}, 1, number_rep);
BigM = blkdiag(MCell{:});

end

